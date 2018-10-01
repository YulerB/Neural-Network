using System;

namespace NeuralNetwork.NetworkModels
{
    public class Gaussian : IActivationFunction
    {
        public double Activate(double x)
		{
            return Math.Exp(Math.Pow(-x, 2d));
        }

        public double Derivative(double x)
        {
            return 2d * x * Math.Exp(Math.Pow(-x, 2d));
        }
    }
}