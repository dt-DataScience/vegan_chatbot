import numpy as np
import pandas as pd
from flask import Flask, flash, request, render_template, jsonify, redirect, session, url_for
from functools import wraps
from haystack.document_store.elasticsearch import ElasticsearchDocumentStore
from haystack.retriever.sparse import ElasticsearchRetriever
from haystack.reader.farm import FARMReader
from haystack.pipeline import ExtractiveQAPipeline
from sentence_transformers import SentenceTransformer
from keras.layers import Input, Dense, Embedding, LSTM
from keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics.pairwise import cosine_similarity
import os
import random
import pickle
from transformers import GPT2LMHeadModel, GPT2Tokenizer
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
genmodel = GPT2LMHeadModel.from_pretrained('gpt2')

app = Flask('gaApp')
app.secret_key = "unicornswithpants"
max_length = 500
vocab_size = 10440
num_class = 30
inputs = Input(shape=(max_length, ))
embedding_layer = Embedding(vocab_size, 128, input_length=max_length)(inputs)
x = LSTM(64)(embedding_layer)
x = Dense(32, activation='relu')(x)
predictions = Dense(num_class, activation='softmax')(x)
chat_classifier = Model(inputs=[inputs], outputs=predictions)
chat_classifier.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['acc'])
chat_classifier.load_weights('./models/weights.hdf5')
with open('./models/tokenizer.pickle', 'rb') as tok:
  chat_tokenizer = pickle.load(tok)

def login_required(f):
  @wraps(f)
  def decorated_function(*args, **kwargs):
    if 'username' not in session:
      flash('You must be logged in. Go click on Login in the nav bar.')
      return redirect(url_for('home_page'))
    return f(*args, **kwargs)
  return decorated_function

def connect_elastic_search():
  os.environ["TOKENIZERS_PARALLELISM"] = "false"
  doc_store = ElasticsearchDocumentStore(
    host='localhost',
    username='',
    password='',
    index='vegan'
  )
  retriever = ElasticsearchRetriever(doc_store)
  reader = FARMReader(model_name_or_path='deepset/bert-base-cased-squad2', context_window_size=500)
  qa = ExtractiveQAPipeline(reader=reader, retriever=retriever)
  return qa

def get_similarities(human_answer, bot_answers=None, canned_answer=None):
  human_answer = human_answer.replace('\n', '')
  if bot_answers != None:
    answers = []
    for bot_answer in bot_answers['answers']:
      answers.append(bot_answer['context'])
    answers.append(human_answer)
  else:
    answers = [canned_answer, human_answer]
  sentTransformer = SentenceTransformer('bert-base-nli-mean-tokens')
  embeddings = sentTransformer.encode(answers)
  return np.round(cosine_similarity([embeddings[-1]], embeddings[:-1]), 2)

def clean_bot_answers(bot_answers):
  for bot_answer in bot_answers['answers']:
    first = 0
    last = len(bot_answer['context']) - 1
    dot_end = False
    for char_index in range(len(bot_answer['context'])):
      if bot_answer['context'][char_index].isupper():
            first = char_index
            break
    try:
      dot = bot_answer['context'].rindex('.')
      if dot > first:
            last = dot
            dot_end = True
    except:
      pass
    try:
      point = bot_answer['context'].rindex('!')
      if dot_end == True:
          last = point if point > first and point > last else last
      elif point > first:
        last = point
        dot_end = True
    except:
      pass
    try:
      q = bot_answer['context'].rindex('?')
      if dot_end == True:
        last = q if q > first and q > last else last
      elif q > first:
        last = q
        dot_end = True
    except:
      pass
    bot_answer['context'] = bot_answer['context'][first:last+1].replace(u'\xa0', u' ')
    if bot_answer['context'][0].upper() != bot_answer['context'][0]:
      bot_answer['context'] = "... " + bot_answer['context']
    if bot_answer['context'][-1] not in ['.', '?', '!']:
      bot_answer['context'] += " ..."

  return bot_answers

def match_miss_max(main_sim, sim, bot_answers):
  # returns match: [({dictonary}, float)]
  # returns miss: [({dictionary}, float)]
  # returns max_match: ({dictionary}, float)
  # if the human answer is similar to the bot answer (0.8 and higher) put it in the match
  # if not, put it in the miss
  # first, sort the indices of the similarities so we know which ones are most similar
  idx = np.argpartition(sim, -len(sim))[-len(sim):]
  indices = idx[np.argsort((-sim)[idx])]
  match = []
  miss = []
  if sim[indices[0]] > main_sim:
    max_sim = sim[indices[0]]
    max_answer = bot_answers['answers'][indices[0]]
    start = 1
  else:
    max_sim = main_sim
    max_answer = None
    start = 0
  for i in range(start, len(indices)):
    bot_index = indices[i]
    if sim[bot_index] > 0.8:
      match.append((bot_answers['answers'][bot_index], sim[bot_index]))
    else:
        miss.append((bot_answers['answers'][bot_index], sim[bot_index]))
  return (match, miss, (max_answer, max_sim))

def identify_troll(human_answer):
  with open('./models/troll_identifier.pkl', 'rb') as pickle_in:
    troll = pickle.load(pickle_in)
  is_troll = troll.predict([human_answer])
  return is_troll

@app.route('/')
def home_page():
  if 'username' in session and 'avocados' not in session:
    session['avocados'] = 3
  if request.args.get('avocados'):
    session['avocados'] = int(request.args.get('avocados'))
  return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
  if request.method == 'GET':
    return render_template('partials/loginform.html')
  else:
    user = request.form['username']
    if len(user) > 1:
      session['username'] = user
    return redirect('/')

@app.route('/quiz')
@login_required
def quiz():
  avocados = session['avocados']
  if avocados <= 0:
    flash("You are out of avocados! Our ideas are not in alignment, so we're curious about your data. Please share before continuing.")
    return redirect('/contribute')
  if 'seen_questions' not in session:
    session['seen_questions'] = []
    session['warned'] = False
  qs = pd.read_json('./data/QA.json')
  new_q = False
  seen = session['seen_questions']
  if len(seen) >= qs.shape[0]:
    # todo: flash message notifying that we're starting over
    session['seen_questions'] = []
    seen = []
  while new_q == False:
    random_question = random.randint(0, qs.shape[0]-1)
    if random_question not in seen:
      new_q = True
      seen.append(random_question)
      session['seen_questions'] = seen
  q = qs.iloc[random_question]
  context = {'question': q}
  return render_template('quiz.html', context=context)


@app.route('/about')
def about():
  return render_template('about.html')

@app.route('/trolling')
@login_required
def trolling():
  human_answer = request.args.get('answer')
  is_troll = identify_troll(human_answer)
  if int(is_troll[0]) == 1:
    session['warned'] = True
  return jsonify({"troll": int(is_troll[0])})

@app.route('/similarities')
@login_required
def calculate_similarities():
  human_answer = request.args.get('answer')
  if session['warned'] == True:
    is_troll = identify_troll(human_answer)
    if int(is_troll[0]) == 1 and session['avocados'] > 0:
      avocados = session['avocados']
      avocados -= 1
      session['avocados'] = avocados
      # To do, what if we're out of avocados?
    session['warned'] = False
  # first want to get similarity with the main answer
  asked_question_number = session['seen_questions'][-1]
  qs = pd.read_json('./data/QA.json')
  q_row = qs.iloc[asked_question_number]
  asked = q_row['question']
  canned_answer = q_row['answer']
  main_sim = get_similarities(human_answer, canned_answer=canned_answer)
  qa = connect_elastic_search()
  bot_answers = qa.run(query=asked, top_k_reader=5)
  bot_answers = clean_bot_answers(bot_answers)

  similarities = get_similarities(human_answer, bot_answers=bot_answers )
  match, miss, max_match = match_miss_max(main_sim[0][0], similarities[0], bot_answers)
  context = {
    "asked": asked,
    "canned_answer": canned_answer,
    "human_answer": human_answer,
    "main_sim": main_sim,
    "match": match,
    "max_match": max_match,
    "miss": miss,
    "tag": q_row['tag'].lower()
  }
  return render_template('feedback.html', context=context)

@app.route('/simdev')
def simdev():
  max_score = float(request.args.get('score')) if request.args.get('score') != "" else 0.5
  human_answer = "I love squishy pies"
  # first want to get similarity with the main answer
  asked_question_number = 2
  qs = pd.read_json('./data/QA.json')
  q_row = qs.iloc[asked_question_number]
  asked = q_row['question']
  canned_answer = q_row['answer']
  match = []
  if max_score > 0.8:
    match = [
      (
        {
          "context": "Matching context info 1",
          "meta": {
            "url": "http://www.match1.com"
          }
        },
        0.8
      ),
      (
        {
          "context": "Matching context info 2",
          "meta": {
            "url": "http://www.match2.com"
          }
        },
        0.8
      )
    ]
  miss = [
    (
      {
        "context": "Missing context info 1",
        "meta": {
          "url": "http://www.miss1.com"
        }
      },
      0.3
    ),
    (
      {
        "context": "Missing context info 2",
        "meta": {
          "url": "http://www.miss2.com"
        }
      },
      0.25
    )
  ]
  if max_score > 0.8:
    max_match = (None, max_score)
  else:
    max_match = ({"context": "Best match context info 1",
          "meta": {
            "url": "http://www.match1.com"
          }
        }, max_score)

  context = {
    "asked": asked,
    "canned_answer": canned_answer,
    "human_answer": human_answer,
    "match": match,
    "max_match": max_match,
    "miss": miss,
    "tag": q_row['tag'].lower()
  }
  return render_template('feedback.html', context=context)

@app.route('/chat', methods=['GET', 'POST'])
# @login_required
def chat():
  if request.method == 'GET':
    return render_template('chat.html')
  else:
    chat = request.form['chat']
    if chat.lower() in ['hi', 'hello', "what's up", 'hey', 'whaddup', 'yo', 'hey hey', 'greetings', "how's it going?"]:
      bot_responses = ['Hello to you too!', 'Greetings!', 'Lovely day today, is it not?', f"Hi, {session['username']}!", "What's going on?"]
      context = {"bot_response": random.choice(bot_responses)}
    elif chat.lower() == "quiz me":
      qs = pd.read_json('./data/QA.json')
      random_question = random.randint(0, qs.shape[0]-1)
      q = qs.iloc[random_question]
      session['chat_quiz'] = True
      session['chat_answer'] = q['answer']
      context = {"bot_response": q['question']}
    elif 'chat_quiz' in session and session['chat_quiz'] == True:
      answer = session['chat_answer']
      similarity = get_similarities(chat, canned_answer = answer)
      if similarity > 0.8:
        context = {
          "bot_response": f"Well done! Your answer was similar to mine: {answer}"
        }
      elif similarity > 0.6:
        context = {
          "bot_response": f"Maybe you have a point. Here was my answer: {answer}"
        }
      else:
        context = {
          "bot_response": f"I'm tempted to take away an avocado for that one. Study up, here's the answer I was looking for: {answer}"
        }
      session['chat_quiz'] = False
      session['chat_answer'] = None
    elif chat[-1] == "?":
      qa = connect_elastic_search()
      bot_answers = qa.run(query=chat, top_k_reader=3)
      bot_answers = clean_bot_answers(bot_answers)
      random_choice = random.choice(bot_answers['answers'])
      context = {
        "bot_response": random_choice['context']
      }
    else:
      class_df = pd.read_json('./data/classification_vb.json')
      post_seq = chat_tokenizer.texts_to_sequences([chat])
      post_seq_padded = pad_sequences(post_seq, maxlen=500)
      pred = chat_classifier.predict(post_seq_padded)
      pred_max = np.argmax(pred, axis=1)
      df = class_df[class_df['target'] == pred_max[0]]
      bot_response_seed = df.sample(n=1)['text'].item()[:100]
      inputs = gpt2_tokenizer.encode(bot_response_seed, add_special_tokens=True, return_tensors='pt')
      outputs = genmodel.generate(inputs, max_length=100, do_sample=True, temperature=1.5, top_k=50)
      bot_response = gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)
      try:
        bot_response_last_dot = bot_response.rindex(".")
        context = {
          "bot_response": bot_response[:bot_response_last_dot + 1]
        }
      except:
        context = {
          "bot_response": bot_response
        }
    return render_template('partials/chat.html', context=context)

@app.route('/contribute', methods=['GET', 'POST'])
@login_required
def contribute():
  if request.method == 'GET':
    return render_template('contribute.html')
  else:
    form = request.form
    if 'avocados' not in session:
      session['avocados'] = 0
    avocados = session['avocados']
    if avocados < 0:
      avocados = 1
      flash("Thank you for the contribution! To show our thanks, we've given you an avocado!")
    elif avocados < 3:
      avocados += 1
      flash("Thank you for the contribution! To show our thanks, we've given you an avocado!")
    else:
      flash("Thank you for the contribution! You already have the maximum number of avocados, but we still appreciate it!")
    session['avocados'] = avocados
    print("form data", form)
    return redirect('/contribute')

@app.route('/pacman')
@login_required
def pacman():
  return render_template('pacman.html')

@app.route('/logout')
def logout():
  session.clear()
  return redirect('/')


if __name__ == "__main__":
  app.run(debug=False)
else:
  print("vegan BTW")