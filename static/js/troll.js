(function($){


  $(function(){
    $("#troll_warning").hide();

    $('#quiz_answer_button').click(function(e) {
      e.preventDefault();
      console.log("clicking the answer button");
      var answer = $('#answer').val();
      $.ajax({
        url: '/trolling?answer=' + answer,
        method: "GET"
      })
      .done(function(res){
        console.log('got this res', res.troll);
        if(res.troll == 1) {
          $("#troll_warning").show();
          $("#quiz_answer_button").hide();
          $("#second_chance_button").show();
        }
        else {
          $.ajax({
            url: '/similarities?answer=' + answer,
            method: "GET"
          })
          .done(function(res) {
            var newDoc = document.open("text/html", "replace");
            newDoc.write(res)
            newDoc.close()
          })
          playPacman();
        }
      })
    })

    $('#second_chance_button').click(function(e) {
      e.preventDefault();
      console.log("clicking the for real answer button");
      var answer = $('#answer').val();
      $.ajax({
        url: '/similarities?answer=' + answer,
        method: "GET"
      })
      .done(function(res) {
        window.stop()
        $('#pacmanwait').remove();
        var newDoc = document.open("text/html", "replace");
        newDoc.write(res);
        newDoc.close();
      })
      playPacman();
    })

    // function playPacman() {
    //   var iframeHtml = "<iframe src='/pacman'></iframe>"
    //   $('#pacman').html(iframeHtml)
    // }
  function playPacman() {
    $.ajax({
      url:'/pacman',
      method: "GET"
    })
    .done(function(res){
      var newDoc = document.open("text/html", "replace");
      newDoc.write(res);
      newDoc.close();
    })
    console.log("wacca wacca");
  }

  });
})(jQuery); // end of jQuery name space
