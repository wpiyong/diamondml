using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;

namespace CSVDataPrepare
{
    public class Program
    {
        static void Main(string[] args)
        {
            string destinationFile = "diamond_data.csv";
            string BaseDatasetsRelativePath = @"../../../Data";
            string DataPath = GetAbsolutePath(BaseDatasetsRelativePath);

            // Specify search to match CSV files that will be combined
            string[] filePaths = Directory.GetFiles(DataPath, "*whitediamond*.csv");
            StreamWriter fileDest = new StreamWriter(DataPath + "//" + destinationFile, false);
            string header = string.Format("{0},{1},{2},{3},{4},{5},{6}", "name", "mean lumi", "std lumi", "percentile", "mean hue", "std hue", "label");
            fileDest.WriteLine(header);

            int i;
            for (i = 0; i < filePaths.Length; i++)
            {
                string file = filePaths[i];
                int label = 1;
                if (file.Contains("nonwhite"))
                {
                    label = 0;
                }
                string[] lines = File.ReadAllLines(file);

                lines = lines.Skip(1).ToArray(); // Skip header row for all but first file

                foreach (string line in lines)
                {
                    // rearrange the line
                    string[] words = line.Split(',');
                    // string.Format("{0},{1},{2},{3},{4},{5},{6}", "index", "mean", "std", "percentile", "name", "mean hue", "std hue");
                    double std = double.Parse(words[2]) / 255;
                    double meanHue = double.Parse(words[5]) / 2;
                    string newLine = string.Format("{0},{1},{2},{3},{4},{5},{6}", words[4], words[1], std, words[3], meanHue, words[6], label);
                    fileDest.WriteLine(newLine);
                }
            }

            fileDest.Close();
        }

        public static string GetAbsolutePath(string relativePath)
        {
            FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
            string assemblyFolderPath = _dataRoot.Directory.FullName;

            string fullPath = Path.Combine(assemblyFolderPath, relativePath);

            return fullPath;
        }
    }
}
