namespace NeuralNetwork.NetworkModels
{
    public interface IActivationFunction
    {
        double Activate(double x);
        double Derivative(double x);
    }
}