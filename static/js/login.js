$('#loggedout').click(function(){
  $.ajax({
    url: '/login',
    method: "GET"
  })
  .done(function(res){
    $("#loggingdiv").html(res);
  })
})


