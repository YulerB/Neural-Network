﻿using System;

namespace NeuralNetwork.NetworkModels
{
    public class TanH : IActivationFunction
    {
        public double Activate(double x)
        {
            return (2d / (1d + Math.Exp(x * -2d))) - 1d;
        }

		public double Derivative(double x)
		{
            return 1d - Math.Pow(x, 2d);
        }
    }
}