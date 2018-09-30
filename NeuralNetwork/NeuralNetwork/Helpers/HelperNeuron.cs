using System;

namespace NeuralNetwork.Helpers
{
    public class HelperNeuron
	{
        public HelperNeuron(double bias, double biasDelta, double gradient, Guid id, double value)
        {
            Bias = bias;
            BiasDelta = biasDelta;
            Gradient = gradient;
            Id = id;
            Value = value;
        }

        public Guid Id { get;private set; }
		public double Bias { get; private set; }
		public double BiasDelta { get; private set; }
		public double Gradient { get; private set; }
		public double Value { get; private set; }
	}
}
