using System;

namespace NeuralNetwork.NetworkModels
{
    public class Sinusoid : IActivationFunction
    {
        public double Activate(double x)
		{
            return Math.Sin(x);
        }

		public double Derivative(double x)
		{
            return Math.Cos(x);
        }
    }
}