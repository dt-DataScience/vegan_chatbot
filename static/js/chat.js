$('#chat_button').click(function(e){
  e.preventDefault();
  chat = $('#chat').val();
  username= $('#loggedin').text()
  username = username.substring(3, username.length-1)
  $('#chat_history').prepend("<p>" + username + ": " + chat + "</p>")
  $('#chat').val('');
  var load_messages;
  $.ajax({
    url: '/chat',
    method: 'POST',
    data: {'chat': chat}
  })
  .done(function(res){
    $('#chat_history').prepend(res);
    $('#loading_messages').html("");
    clearInterval(load_messages);
  })
  .fail(function(res){
    alert("error")
  })
  $('#loading_messages').html("<p>Hang on....</p>")
  messages = ["Please be patient, we're thinking", "How many vegans does it take to change a light bulb?", "Two. One to change it and one to check for animal ingredients", "Figuring out cosine similarity, probably", "It's possible that elastic search is throwing a fit", "What does a vegan zombie eat?", "GRAAAAIIIINNNNS", "Thanks for sticking around, it'll be worth it, maybe", "Want to know what's going on behind the scenes?", "First we figure out if you're asking a question", "If you are, we go through our elasticsearch database to construct a reasonable answwer", "If you're answering a question, we determine the cosine similarity to our answer", "If you're just talking, we have more work to do", "We categorize your speech to determine what you're talking about", "Then we look through all our data to see what we have to say abou that topic", "Then we randomly select a sentence starter and pass it to GPT2", "GPT2 uses that sentence starter and its knowledge of human speech to create a novel sentence", "Sometimes we make sense, sometimes we don't!"]
  i = 0;
  load_messages = setInterval(function(){
    $('#loading_messages').html("<p>" + messages[i] + "</p>")
    if(i == messages.length - 1) {
      i = 0;
    }
    else {
      i += 1;
    }
  }, 3000)

})