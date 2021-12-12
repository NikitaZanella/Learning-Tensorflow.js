import * as tf from "@tensorflow/tfjs";
import * as tfvis from "@tensorflow/tfjs-vis";
import { useEffect } from "react";

function App() {

  async function plot(pointsArray, featureName) {
    tfvis.render.scatterplot(
      {name: `${featureName} vs House Price`},
      {values: [pointsArray], series: ['Original']},
      {
        xLabel: featureName,
        yLabel: 'Price'
      }
      )
  }

  function normalise(tensor) {
    const min = tensor.min();
    const max = tensor.max();

    const normalisedTensor = tensor.sub(min).div(max.sub(min));

    return normalisedTensor;
  } 

  function createModel() {
    const model = tf.sequential();

    /*
    Create a dense model that means that each node (neuron)
    receive every inputs from all the previous nodes.
    */

    model.add(tf.layers.dense({
      units: 1,  // Numbers of nodes that would be used in the model
      useBias: true, // The bias allow the regression line to cross the y axis for better training.
      activation: 'linear', // Activation function - Non needed in this model
      inputDim: 1,

    }))

    return model;
  }
   
  useEffect(async () => {

    // Import data from CSV
    const dataset = tf.data.csv('/data.csv');


    // Extract x and y values to plot
    const pointsDataset = dataset.map(record => ({
      x: record.sqft_living,
      y: record.price,
    }));

    const points = await pointsDataset.toArray();
    if (points.length % 2 !== 0) { // If odd number fo elements
      points.pop(); // Remove one element

      /* 
      This process is needed to split our data into 2 different tensors.
      If points.lenght is an odd number, we can't divide it by 2 keeping an integer.
      */

    }
    tf.util.shuffle(points);
    plot(points, 'sqft')

    // Features (inputs)
    const featureValues = points.map(p => p.x);
    const featureTensor = tf.tensor2d(featureValues, [featureValues.length, 1]);

    // Labels (Outputs)
    const labelValues = points.map(p => p.y);
    const labelTensor = tf.tensor2d(labelValues, [labelValues.length, 1])

    // Normalise features and labels

    const normalisedFeature = normalise(featureTensor);
    const normalisedLabel = normalise(labelTensor);

   const [trainingFeatureTensor, testingFeatureTensor] = tf.split(normalisedFeature, 2);
   const [trainingLabelTensor, testingLabelTensor] = tf.split(normalisedLabel, 2);

   const model = createModel();

  // How to inspect a model
  // model.summary(); // This will print a summary of the model, including trainable params count.

    tfvis.show.modelSummary({name: 'Model summary'}, model);
    const layer = model.getLayer(undefined, 0);
    tfvis.show.layer({name: 'Layer 1'}, layer);
    
    // Compile the model adding loss, optimizer
    // create an optimizer
    const optimizer = tf.train.sgd(0.1);
    model.compile({
      loss: 'meanSquaredError', // A median squared error
      optimizer, // In this case I've used an sgd aka stochastic gradient descent with a learning reate of 0.1
    })

  }, [])

  return (
    <div className="App">
      <p>Hello Tensor!</p>
    </div>
  );

}

export default App;
