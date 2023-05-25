//jshint esversion:6

const express = require("express");
const bodyParser = require("body-parser");

const app = express();

app.use(bodyParser.urlencoded({extended: true}))
app.use(express.static("public"))
app.set('view engine', 'ejs');

var items = ["Buy Food", "Cook Food"]

app.get("/", function(req, res) {
  var today = new Date();
  var day = ""

  var options = {
    weekday: "long",
    day: "numeric",
    month: "long"
  };

  day = today.toLocaleDateString("en-US", options);

  res.render('list', {
    CURR_DAY: day,
    NEW_LIST_ITEMS: items
  });

});

app.post("/", function(req, res){
  items.push(req.body.newItem);
  res.redirect("/");
})

app.listen(3000, function() {
  console.log("Server started on port 3000.");
});
