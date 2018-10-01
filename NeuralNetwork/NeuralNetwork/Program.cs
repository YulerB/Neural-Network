using System;
using System.Collections.Generic;
using System.Linq;
using NeuralNetwork.Helpers;
using NeuralNetwork.NetworkModels;

namespace NeuralNetwork
{
	internal class Program
	{

        #region -- Main --
        [STAThread]
        private static void Main()
        {
            MakeNetworkBinary2OR();
            MakeNetworkBinary3ToDecimal();
            MakeNetworkBinaryDigitAdd();
            MakeNetworkForBinary4ToDecimal();

            Console.ReadKey();
        }

        private static void MakeNetworkBinary2OR()
        {
            Network _network = new Network(2, new int[] { 2 }, new IActivationFunction[] { new Sigmoid() }, 1, new Sigmoid());
            var dataSet = new List<DataSet>{
                new DataSet(new double[]{0,0}, new double[]{0}),
                new DataSet(new double[]{0,1}, new double[]{1}),
                new DataSet(new double[]{1,0}, new double[]{1}),
                new DataSet(new double[]{1,1}, new double[]{0})
            };
            double minErrorRate = 0.03;
            var epochs = _network.Train(dataSet, minErrorRate);

            Console.WriteLine($"Training Epochs: {epochs} to achieve an error rate of {minErrorRate} using {_network.HiddenLayersCount} hidden layer(s) with {_network.HiddenLayerNeuronCount} neurons");

            foreach (var set in dataSet)
            {
                var output = _network.Compute(new double[] { set.Values[0], set.Values[1] });
                Console.WriteLine($"({set.Values[0]},{set.Values[1]}) = {output[0]} rounded to {Math.Round(output[0], 0)}, expected {set.Targets[0]}");
            }

            ExportHelper.ExportNetwork(_network);

            ImportHelper.ImportNetwork();

            foreach (var set in dataSet)
            {
                var output = _network.Compute(new double[] { set.Values[0], set.Values[1] });
                Console.WriteLine($"({set.Values[0]},{set.Values[1]}) = {output[0]} rounded to {Math.Round(output[0], 0)}, expected {set.Targets[0]}");
            }
        }

        private static void MakeNetworkBinary3ToDecimal()
        {
            var _network = new Network(3, new int[] { 5 }, new IActivationFunction[] { new Sigmoid() }, 1, new Sigmoid());
            var dataSet = new List<DataSet>{
                new DataSet(new double[]{0, 0, 0}, new double[]{0}),
                new DataSet(new double[]{0, 0, 1}, new double[]{0.1}),
                new DataSet(new double[]{0, 1, 0}, new double[]{0.2}),
                new DataSet(new double[]{0, 1, 1}, new double[]{0.3}),
                new DataSet(new double[]{1, 0, 0}, new double[]{0.4}),
                new DataSet(new double[]{1, 0, 1}, new double[]{0.5}),
                new DataSet(new double[]{1, 1, 0}, new double[]{0.6}),
                new DataSet(new double[]{1, 1, 1}, new double[]{0.7})
            };
            var minErrorRate = 0.003;
            var epochs = _network.Train(dataSet, minErrorRate);

            Console.WriteLine($"Training Epochs: {epochs} to achieve an error rate of {minErrorRate} using {_network.HiddenLayersCount} hidden layer(s) with {_network.HiddenLayerNeuronCount} neurons");

            foreach (var set in dataSet)
            {
                var output = _network.Compute(new double[] { set.Values[0], set.Values[1], set.Values[2] });
                Console.WriteLine($"({set.Values[0]},{set.Values[1]},{set.Values[2]}) = {output[0]} rounded to {Math.Round(output[0], 1)}, expected {set.Targets[0]}");
            }
        }

        private static void MakeNetworkBinaryDigitAdd()
        {
            var _network = new Network(4, new int[] { 8 }, new IActivationFunction[] { new Sigmoid() }, 1, new Sigmoid());
            var dataSet = new List<DataSet>{
                new DataSet(new double[]{ 0, 0, 0, 0 }, new double[]{0.0}),
                new DataSet(new double[]{ 0, 0, 0, 1 }, new double[]{0.1}),
                new DataSet(new double[]{ 0, 0, 1, 0 }, new double[]{0.1}),
                new DataSet(new double[]{ 0, 0, 1, 1 }, new double[]{0.2}),
                new DataSet(new double[]{ 0, 1, 0, 0 }, new double[]{0.1}),
                new DataSet(new double[]{ 0, 1, 0, 1 }, new double[]{0.2}),
                new DataSet(new double[]{ 0, 1, 1, 0 }, new double[]{0.2}),
                new DataSet(new double[]{ 0, 1, 1, 1 }, new double[]{0.3}),
                new DataSet(new double[]{ 1, 0, 0, 0 }, new double[]{0.1}),
                new DataSet(new double[]{ 1, 0, 0, 1 }, new double[]{0.2}),
                new DataSet(new double[]{ 1, 0, 1, 0 }, new double[]{0.2}),
                new DataSet(new double[]{ 1, 0, 1, 1 }, new double[]{0.3}),
                new DataSet(new double[]{ 1, 1, 0, 0 }, new double[]{0.2}),
                new DataSet(new double[]{ 1, 1, 0, 1 }, new double[]{0.3}),
                new DataSet(new double[]{ 1, 1, 1, 0 }, new double[]{0.3}),
                new DataSet(new double[]{ 1, 1, 1, 1 }, new double[]{0.4})
            };
            var minErrorRate = 0.0003;
            var epochs = _network.Train(dataSet, minErrorRate);

            Console.WriteLine($"Training Epochs: {epochs} to achieve an error rate of {minErrorRate} using {_network.HiddenLayersCount} hidden layer(s) with {_network.HiddenLayerNeuronCount} neurons");

            foreach (var set in dataSet)
            {
                var output = _network.Compute(new double[] { set.Values[0], set.Values[1], set.Values[2], set.Values[3] });
                Console.WriteLine($"({set.Values[0]},{set.Values[1]},{set.Values[2]},{set.Values[3]}) = {output[0]} rounded to {Math.Round(output[0], 2)}, expected {set.Targets[0]}");
            }
        }

        private static void MakeNetworkForBinary4ToDecimal()
        {
            var _network = new Network(4, new int[] { 8 }, new IActivationFunction[] { new Sigmoid() }, 1, new Sigmoid());
            var dataSet = new List<DataSet>{
                new DataSet(new double[]{ 0, 0, 0, 0 }, new double[]{0.00}),
                new DataSet(new double[]{ 0, 0, 0, 1 }, new double[]{0.01}),
                new DataSet(new double[]{ 0, 0, 1, 0 }, new double[]{0.02}),
                new DataSet(new double[]{ 0, 0, 1, 1 }, new double[]{0.03}),
                new DataSet(new double[]{ 0, 1, 0, 0 }, new double[]{0.04}),
                new DataSet(new double[]{ 0, 1, 0, 1 }, new double[]{0.05}),
                new DataSet(new double[]{ 0, 1, 1, 0 }, new double[]{0.06}),
                new DataSet(new double[]{ 0, 1, 1, 1 }, new double[]{0.07}),
                new DataSet(new double[]{ 1, 0, 0, 0 }, new double[]{0.08}),
                new DataSet(new double[]{ 1, 0, 0, 1 }, new double[]{0.09}),
                new DataSet(new double[]{ 1, 0, 1, 0 }, new double[]{0.10}),
                new DataSet(new double[]{ 1, 0, 1, 1 }, new double[]{0.11}),
                new DataSet(new double[]{ 1, 1, 0, 0 }, new double[]{0.12}),
                new DataSet(new double[]{ 1, 1, 0, 1 }, new double[]{0.13}),
                new DataSet(new double[]{ 1, 1, 1, 0 }, new double[]{0.14}),
                new DataSet(new double[]{ 1, 1, 1, 1 }, new double[]{0.15})
            };
            var minErrorRate = 0.0003;
            var epochs = _network.Train(dataSet, minErrorRate);

            Console.WriteLine($"Training Epochs: {epochs} to achieve an error rate of {minErrorRate} using {_network.HiddenLayersCount} hidden layer(s) with {_network.HiddenLayerNeuronCount} neurons");

            foreach (var set in dataSet)
            {
                var output = _network.Compute(new double[] { set.Values[0], set.Values[1], set.Values[2], set.Values[3] });
                Console.WriteLine($"({set.Values[0]},{set.Values[1]},{set.Values[2]},{set.Values[3]}) = {output[0]} rounded to {Math.Round(output[0], 2)}, expected {set.Targets[0]}");
            }
        }
        #endregion

    }
}
