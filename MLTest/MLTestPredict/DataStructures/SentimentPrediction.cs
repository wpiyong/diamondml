﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.Data;

namespace MLTestPredict.DataStructures
{
    public class SentimentPrediction
    {
        // ColumnName attribute is used to change the column name from
        // its default value, which is the name of the field.
        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }

        // No need to specify ColumnName attribute, because the field
        // name "Probability" is the column name we want.
        public float Probability { get; set; }

        public float Score { get; set; }
    }
}
