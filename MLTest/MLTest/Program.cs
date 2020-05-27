using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Runtime;
using MLTest.DataStructures;
using MLTest.Common;
using static Microsoft.ML.DataOperationsCatalog;

namespace MLTest
{
    public enum MLModel
    {
        MLSentiment,
        MLDiamond
    };

    internal static class Program
    {
        private static readonly string BaseDatasetsRelativePath = @"../../../../Data";
        private static readonly string BaseModelsRelativePath = @"../../../../MLModels";

        private static string DataRelativePath = "";
        private static string DataPath = "";
        private static string ModelRelativePath = "";
        private static string ModelPath = "";

        private static MLModel mlModel = MLModel.MLDiamond;

        static void Main(string[] args)
        {
            // Create MLContext to be shared across the model creation work flow objects 
            // Set a random seed for repeatable/deterministic results across multiple trainings.
            try
            {
                if (mlModel == MLModel.MLSentiment)
                {
                    var mlContext = new MLContext();

                    DataRelativePath = $"{BaseDatasetsRelativePath}/wikiDetoxAnnotated40kRows.tsv";
                    DataPath = GetAbsolutePath(DataRelativePath);
                    ModelRelativePath = $"{BaseModelsRelativePath}/SentimentModel.zip";
                    ModelPath = GetAbsolutePath(ModelRelativePath);

                    // STEP 1: Common data loading configuration
                    IDataView dataView = mlContext.Data.LoadFromTextFile<SentimentIssue>(DataPath, hasHeader: true);

                    DataOperationsCatalog.TrainTestData trainTestSplit = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);
                    IDataView trainingData = trainTestSplit.TrainSet;
                    IDataView testData = trainTestSplit.TestSet;

                    // STEP 2: Common data process configuration with pipeline data transformations          
                    var dataProcessPipeline = mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(SentimentIssue.Text));

                    // STEP 3: Set the training algorithm, then create and config the modelBuilder                            
                    var trainer = mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features");
                    var trainingPipeline = dataProcessPipeline.Append(trainer);

                    // STEP 4: Train the model fitting to the DataSet
                    ITransformer trainedModel = trainingPipeline.Fit(trainingData);

                    // STEP 5: Evaluate the model and show accuracy stats
                    var predictions = trainedModel.Transform(testData);
                    var metrics = mlContext.BinaryClassification.Evaluate(data: predictions, labelColumnName: "Label", scoreColumnName: "Score");

                    ConsoleHelper.PrintBinaryClassificationMetrics(trainer.ToString(), metrics);

                    // STEP 6: Save/persist the trained model to a .ZIP file
                    mlContext.Model.Save(trainedModel, trainingData.Schema, ModelPath);

                    Console.WriteLine("The model is saved to {0}", ModelPath);

                    // TRY IT: Make a single test prediction, loading the model from .ZIP file
                    SentimentIssue sampleStatement = new SentimentIssue { Text = "I love this movie!" };

                    // Create prediction engine related to the loaded trained model
                    var predEngine = mlContext.Model.CreatePredictionEngine<SentimentIssue, SentimentPrediction>(trainedModel);

                    // Score
                    var resultprediction = predEngine.Predict(sampleStatement);

                    Console.WriteLine($"=============== Single Prediction  ===============");
                    Console.WriteLine($"Text: {sampleStatement.Text} | Prediction: {(Convert.ToBoolean(resultprediction.Prediction) ? "Toxic" : "Non Toxic")} sentiment | Probability of being toxic: {resultprediction.Probability} ");
                    Console.WriteLine($"================End of Process.Hit any key to exit==================================");
                } else if(mlModel == MLModel.MLDiamond)
                {
                    DiamondML();
                }
            } catch(Exception ex)
            {
                Console.WriteLine(ex.ToString());
            }

            Console.ReadLine();
        }

        private static void DiamondML()
        {
            var mlContext = new MLContext();

            DataRelativePath = $"{BaseDatasetsRelativePath}/diamond_data.csv";
            DataPath = GetAbsolutePath(DataRelativePath);
            ModelRelativePath = $"{BaseModelsRelativePath}/whitediamondModel.zip";
            ModelPath = GetAbsolutePath(ModelRelativePath);

            // STEP 1: Common data loading configuration
            IDataView dataView = mlContext.Data.LoadFromTextFile<DiamondData>(DataPath, separatorChar: ',', hasHeader: true);

            DataOperationsCatalog.TrainTestData trainTestSplit = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);
            IDataView trainingData = trainTestSplit.TrainSet;
            IDataView testData = trainTestSplit.TestSet;

            // STEP 2: Common data process configuration with pipeline data transformations          
            //var dataProcessPipeline = mlContext.Transforms.CopyColumns(outputColumnName: "Features", inputColumnName: nameof(DiamondData.LumiStdHueMean));
            var dataProcessPipeline = mlContext.Transforms.Concatenate("Features", "LumiStd", "HueMean");

            // STEP 3: Set the training algorithm, then create and config the modelBuilder                            
            var trainer = mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features");
            var trainingPipeline = dataProcessPipeline.Append(trainer);

            // STEP 4: Train the model fitting to the DataSet
            ITransformer trainedModel = trainingPipeline.Fit(trainingData);

            // STEP 5: Evaluate the model and show accuracy stats
            var predictions = trainedModel.Transform(testData);
            var metrics = mlContext.BinaryClassification.Evaluate(data: predictions, labelColumnName: "Label", scoreColumnName: "Score");

            ConsoleHelper.PrintBinaryClassificationMetrics(trainer.ToString(), metrics);
            
            // STEP 6: Save/persist the trained model to a .ZIP file
            mlContext.Model.Save(trainedModel, trainingData.Schema, ModelPath);

            Console.WriteLine("The model is saved to {0}", ModelPath);

            // TRY IT: Make a single test prediction, loading the model from .ZIP file
            DiamondData sampleDiamond = new DiamondData { LumiStd = 0.1169f, HueMean =0.13599f };

            // Create prediction engine related to the loaded trained model
            var predEngine = mlContext.Model.CreatePredictionEngine<DiamondData, DiamondPrediction>(trainedModel);

            // Score
            var resultprediction = predEngine.Predict(sampleDiamond);

            Console.WriteLine($"=============== Single Prediction  ===============");
            Console.WriteLine($"{sampleDiamond.LumiStd}, {sampleDiamond.HueMean} | Prediction: {(Convert.ToBoolean(resultprediction.Prediction) ? "white diamond" : "Non white diamond")} Probability of being white diamond: {resultprediction.Probability} ");
            Console.WriteLine($"================End of Process.Hit any key to exit==================================");
            
        }

        private static string GetAbsolutePath(string relativePath)
        {
            FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
            string assemblyFolderPath = _dataRoot.Directory.FullName;

            string fullPath = Path.Combine(assemblyFolderPath, relativePath);

            return fullPath;
        }
    }
}
