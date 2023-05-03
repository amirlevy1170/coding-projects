var buttonColours =["red", "blue", "green", "yellow"];
var gamePattern = [];
var userClickedPattern = [];
var level = 0;
var started = false;

function nextSequence(){
  userClickedPattern = [];
  $("h1").text("LEVEL " + level);
  level = level + 1;
  var randomNumber = Math.floor(Math.random() * 4);
  randomChosenColour = buttonColours[randomNumber];
  gamePattern.push(randomChosenColour);
  setTimeout(function (val) {
  fullAnimate(val);
}, 1000, gamePattern[gamePattern.length - 1]);
  }
$(".btn").click(function() {
    var color = this.id;
    userClickedPattern.push(color);
    fullAnimate(color);
    checkAnswer(1);
});

function playSound(name){
  var audio = new Audio("sounds/" + name + ".mp3");
  audio.play();
}

function pressAnimation(name){
  var idColor = "#" + name;
  $(idColor).fadeOut(50).fadeIn(50);
}
function fullAnimate(name){
  pressAnimation(name);
  playSound(name);
}
function checkAnswer(currentLevel){
  var last_index = userClickedPattern.length - 1;
  if (userClickedPattern[last_index] != gamePattern[last_index]){
    $("body").addClass("game-over");
    setTimeout(function () {
    $("body").removeClass("game-over")
  }, 200);
  $("h1").text("Game Over, Press Any Key to Restart ");
  started = false;
  gamePattern = [];
  level = 0;
  return;
}
  if ((last_index + 1) == (gamePattern.length)){
    setTimeout(function () {
    nextSequence();
  }, 1000);
  }
}
$("body").keypress(function() {
  if (!started) {
    $("#level-title").text("Level " + level);
    nextSequence();
    started = true;
  }
})
