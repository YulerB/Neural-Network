using System;

namespace NeuralNetwork.Helpers
{
    public class HelperNeuron
	{
        public HelperNeuron() { }
        public HelperNeuron(double bias, double biasDelta, double gradient, Guid id, double value, string activationFunctionType)
        {
            Bias = bias;
            BiasDelta = biasDelta;
            Gradient = gradient;
            Id = id;
            Value = value;
            ActivationFunctionType = activationFunctionType;
        }
        public string ActivationFunctionType { get; set; }
        public Guid Id { get; set; }
		public double Bias { get;  set; }
		public double BiasDelta { get;  set; }
		public double Gradient { get;  set; }
		public double Value { get;  set; }
	}
}
