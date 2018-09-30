using System;

namespace NeuralNetwork.Helpers
{
    public class HelperNeuron
	{
        public HelperNeuron() { }
        public HelperNeuron(double bias, double biasDelta, double gradient, Guid id, double value)
        {
            Bias = bias;
            BiasDelta = biasDelta;
            Gradient = gradient;
            Id = id;
            Value = value;
        }

        public Guid Id { get; set; }
		public double Bias { get;  set; }
		public double BiasDelta { get;  set; }
		public double Gradient { get;  set; }
		public double Value { get;  set; }
	}
}
