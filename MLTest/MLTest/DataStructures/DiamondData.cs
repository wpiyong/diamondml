using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLTest.DataStructures
{
    // string.Format("{0},{1},{2},{3},{4},{5},{6}", "name", "mean lumi", "std lumi", "percentile", "mean hue", "std hue", "label");
    public class DiamondData
    {
        [LoadColumn(6)]
        public bool Label { get; set; }
        [LoadColumn(new int[] { 2, 4 })]
        [VectorType(2)]
        public float[] LumiStdHueMean { get; set; }
    }

    public class DiamondPrediction
    {
        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }

        // No need to specify ColumnName attribute, because the field
        // name "Probability" is the column name we want.
        public float Probability { get; set; }

        public float Score { get; set; }
    }
}
