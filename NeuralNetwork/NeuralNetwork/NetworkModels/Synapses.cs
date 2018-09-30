using NeuralNetwork.Helpers;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetwork.NetworkModels
{
    public class Synapses : List<Synapse> {

        public double CalculateValue()
        {
            return this.Sum(_=> _.CalculateValue());
        }

        public IEnumerable<HelperSynapse> ToHelperSynapses()
        {
            return this.Select(_ => _.ToHelperSynapse());
        }
    }
}