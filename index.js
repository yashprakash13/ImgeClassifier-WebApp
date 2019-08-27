const MODEL_URL =
  "https://raw.githubusercontent.com/yashprakash13/ML-Models/master/Mobilenet_Models/tensorflowjs_model.pb";
const WEIGHTS_URL =
  "https://raw.githubusercontent.com/yashprakash13/ML-Models/master/Mobilenet_Models/weights_manifest.json";
let model;
let IMAGENET_CLASSES = [];
let offset = tf.scalar(128);
async function loadModelAndClasses() {
  $.getJSON(
    "https://raw.githubusercontent.com/yashprakash13/ML-Models/master/Mobilenet_Models/imagenet_classes.json",
    function (data) {
      $.each(data, function (key, val) {
        IMAGENET_CLASSES.push(val);
      });
    }
  );
  model = await tf.loadFrozenModel(MODEL_URL, WEIGHTS_URL);
  $(".loadingDiv").hide();
  $("#inputImage").attr("disabled", false);
}
loadModelAndClasses();

function readURL(input) {
  if (input.files && input.files[0]) {
    var reader = new FileReader();

    reader.onload = function (e) {
      $("#imageSrc")
        .attr("src", e.target.result)
        .width(224)
        .height(224);
    };

    reader.readAsDataURL(input.files[0]);

    reader.onloadend = async function () {
      let imageData = document.getElementById("imageSrc");

      let pixels = tf.fromPixels(imageData);
      let expPixels = pixels.resizeNearestNeighbor([224, 224]).toFloat().sub(offset).div(offset).expandDims();

      const output = await model.predict(expPixels);
      const predictions = Array.from(output.dataSync())
        .map(function (p, i) {
          return {
            probabilty: p,
            classname: IMAGENET_CLASSES[i]
          };
        })
        .sort((a, b) => b.probabilty - a.probabilty)
        .slice(0, 5);

      var html = "";
      for (let i = 0; i < 5; i++) {
        html += "<li>" + predictions[i].classname + "</li>";
      }
      $(".predictionList").html(html);

      pixels.dispose();
      expPixels.dispose();
      output.dispose();
    };
  }
}
