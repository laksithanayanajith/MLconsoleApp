// See https://aka.ms/new-console-template for more information
using Microsoft.ML;

Console.WriteLine("Prodecting Housing Values");

var context = new MLContext();

var data = context.Data.LoadFromTextFile<HousingData>("./File/housing.csv", hasHeader: true, separatorChar: ',');

var split = context.Data.TrainTestSplit(data, testFraction: 0.2);

var features = split.TrainSet.Schema.Select(col => col.Name).Where(colName => colName != "Label" & colName != "OceanProximity").ToArray();

var pipeline = context.Transforms.Text.FeaturizeText("Text", "OceanProximity")
    .Append(context.Transforms.Concatenate("Features", features))
    .Append(context.Transforms.Concatenate("Feature", "Features", "Text"))
    .Append(context.Regression.Trainers.FastTreeTweedie());

var model = pipeline.Fit(split.TrainSet);

var predictions = model.Transform(split.TestSet);

var matrics = context.Regression.Evaluate(predictions);

if (matrics is not null)
{
    Console.WriteLine("Model works without issue!");

    Console.WriteLine("R^2 is " + matrics.RSquared);

    if (matrics.RSquared > 0.6)
    {
        Console.WriteLine("Best Algorithm!");
    }
    else
    {
        Console.WriteLine("This algorithm not best fit!");
    }
}