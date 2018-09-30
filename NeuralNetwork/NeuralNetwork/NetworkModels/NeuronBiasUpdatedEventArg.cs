using System;

namespace NeuralNetwork.NetworkModels
{
    public class NeuronBiasUpdatedEventArg : EventArgs
    {
        public NeuronBiasUpdatedEventArg(double learnRate, double momentum, double gradient)
        {
            this.LearnRate = learnRate;
            this.Momentum = momentum;
            this.Gradient = gradient;
        }
        public double LearnRate { get; private set; }
        public double Momentum { get; private set; }
        public double Gradient { get; private set; }
    }
}