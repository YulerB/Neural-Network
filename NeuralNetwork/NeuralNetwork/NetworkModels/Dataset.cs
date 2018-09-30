namespace NeuralNetwork.NetworkModels
{
	public class DataSet
	{
		public double[] Values { get; private set; }
		public double[] Targets { get; private set; }

		public DataSet(double[] values, double[] targets)
		{
			Values = values;
			Targets = targets;
		}
	}
}