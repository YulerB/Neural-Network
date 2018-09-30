using System;

namespace NeuralNetwork.NetworkModels
{
    /// <summary>
    /// 0 to 1
    /// </summary>
    public class Sigmoid : IActivationFunction
    {
		public double Activate(double x)
		{
			return 1d / (1d + Math.Exp(-x));
		}

		public double Derivative(double x)
		{
            return x * (1d - x);
		}
	}
}