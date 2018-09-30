using System.Collections.Generic;
using System.Linq;

namespace NeuralNetwork.NetworkModels
{
    public class NeuralLayer : List<Neuron>
    {
        public NeuralLayer() : base() { }
        public NeuralLayer(IEnumerable<Neuron> neurons) : base(neurons) {}
        public NeuralLayer(int inputSize) : base(Enumerable.Range(0, inputSize).Select(_ => new Neuron())) { }
        public NeuralLayer(int inputSize, NeuralLayer InputLayer) : base (Enumerable.Range(0, inputSize).Select(_ => new Neuron(InputLayer))){}
        internal void SetInputs(double[] inputs)
        {
            int i = 0;
            base.ForEach(a => a.SetInput(inputs[i++]));
        }
        public IEnumerable<double> GetValues()
        {
            return this.Select(a => a.GetValue());
        }
        public void CalculateValues()
        {
            base.ForEach(b => b.CalculateValue());
        }
        public void CalculateGradients()
        {
            base.ForEach(b => b.CalculateGradient());
        }
        public void UpdateBias(double learnRate, double momentum)
        {
            base.ForEach(b => b.UpdateBias(learnRate, momentum));
        }
        public void CalculateGradients(double[] targets)
        {
            int i = 0;
            base.ForEach(a => a.CalculateGradient(targets[i++]));
        }
    }
}