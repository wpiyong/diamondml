using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML;
using MLTestPredict.DataStructures;

namespace MLTestPredict
{
    public class Program
    {
        static void Main(string[] args)
        {
            try
            {
                // Create MLContext
                MLContext mlContext = new MLContext();

                // Load Trained Model
                DataViewSchema predictionPipelineSchema;
                ITransformer predictionPipeline = mlContext.Model.Load("../../../MLModels/SentimentModel.zip", out predictionPipelineSchema);

                // Create PredictionEngines
                PredictionEngine<SentimentIssue, SentimentPrediction> predictionEngine = mlContext.Model.CreatePredictionEngine<SentimentIssue, SentimentPrediction>(predictionPipeline);

                // Input Data (single)
                SentimentIssue inputData = new SentimentIssue
                {
                    Text = "I love this movie!"
                };

                // Get Prediction
                SentimentPrediction prediction = predictionEngine.Predict(inputData);

                Console.WriteLine($"=============== Single Prediction  ===============");
                Console.WriteLine($"Text: {inputData.Text} | Prediction: {(Convert.ToBoolean(prediction.Prediction) ? "Toxic" : "Non Toxic")} sentiment | Probability of being toxic: {prediction.Probability} ");
                Console.WriteLine($"================End of Process.Hit any key to exit==================================");

                // input data multiple 
                SentimentIssue[] inputArray = new SentimentIssue[]
                {
                    new SentimentIssue
                    {
                        Text = "I love this movie!"
                    },
                    new SentimentIssue
                    {
                        Text = "Fucking good!"
                    },
                };

                //Load Data
                IDataView data = mlContext.Data.LoadFromEnumerable<SentimentIssue>(inputArray);

                // Predicted Data
                IDataView predictions = predictionPipeline.Transform(data);

                // Create an IEnumerable of prediction objects from IDataView
                IEnumerable<SentimentPrediction> dataEnumerable =
                    mlContext.Data.CreateEnumerable<SentimentPrediction>(predictions, reuseRowObject: true);

                // Iterate over each row
                for(int i = 0; i < inputArray.Length; i++)
                {
                    string input = inputArray[i].Text;
                    SentimentPrediction item = dataEnumerable.ElementAt(i);
                    // display
                    Console.WriteLine($"Text: {input} | Prediction: {(Convert.ToBoolean(item.Prediction) ? "Toxic" : "Non Toxic")} sentiment | Probability of being toxic: {item.Probability} ");
                }
            }
            catch(Exception ex)
            {
                Console.WriteLine(ex.ToString());
            }

            Console.ReadLine();
        }
    }
}
