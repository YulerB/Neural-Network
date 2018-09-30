using System;

namespace NeuralNetwork.NetworkModels
{
    /// <summary>
    /// -1 to 1
    /// </summary>
    public class TanH : IActivationFunction
    {
		public double Activate(double x)
		{
            return Math.Tanh(x);
        }

		public double Derivative(double x)
		{
            return 1d / Math.Pow(Math.Cosh(x), 2d);
        }
    }
}