using System;

namespace NeuralNetwork.NetworkModels
{
    public class SoftPlus : IActivationFunction
    {
        public double Activate(double x)
		{
            return Math.Log(1d + Math.Exp(x));
        }

		public double Derivative(double x)
		{
            return 1d / (1d + Math.Exp(-x));
        }
    }
}