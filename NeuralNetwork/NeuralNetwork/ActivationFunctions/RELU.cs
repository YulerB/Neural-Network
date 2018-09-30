using System;

namespace NeuralNetwork.NetworkModels
{
    /// <summary>
    /// 0 to any positive number, doesn't really work, as it sometimes cannot ever find the answer.
    /// </summary>
    public class RELU : IActivationFunction
    {
        public double Activate(double x)
		{
            return Math.Max(0d, x);
        }

		public double Derivative(double x)
		{
            return x <= 0d ? 0d : 1d;
        }
    }
}