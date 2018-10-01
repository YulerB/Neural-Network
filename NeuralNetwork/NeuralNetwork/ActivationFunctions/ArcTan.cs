using System;

namespace NeuralNetwork.NetworkModels
{
    public class ArcTan : IActivationFunction
    {
        public double Activate(double x)
        {
            return Math.Tan(x) * -1d;
        }

		public double Derivative(double x)
		{
            return 1d / (Math.Pow(x, 2d) + 1d);
        }
    }
}