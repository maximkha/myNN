using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace myNN
{
    public static class NN
    {
        public static neuralNet fromXml(string path)
        {
			FileStream fs = new FileStream(path + ".xml", FileMode.OpenOrCreate);
			System.Xml.Serialization.XmlSerializer s = new System.Xml.Serialization.XmlSerializer(typeof(neuralNet));
            return (neuralNet) s.Deserialize(fs);
        }

        public enum activationFunction
        {
            input,
            sigmoid,
            relu,
            tanh
        }

        public enum learningMode
        {
            staticLearn,
            restartDecay
        }

        [Serializable]
        public class learningRateDescriptor
        {
            public learningMode learnMode;
            public int epoch;
            public int cycle;
            public double startLearnRate;

            //https://towardsdatascience.com/understanding-learning-rates-and-how-it-improves-performance-in-deep-learning-d0d4059c1c10
            public double learningRate()
            {
                if (learnMode.Equals(learningMode.restartDecay))
                {
                    //Using sin function
                    epoch++;
                    if (epoch >= cycle)
                    {
                        //DEBUG
                        Console.WriteLine("LR");
                        epoch = 0;
                    }
                    return Math.Sin(((Math.PI*epoch)/(cycle*2))+(Math.PI/2)) * startLearnRate;
                }

                return startLearnRate;
            }
        }

        [Serializable]
        public class neuron
        {
            public double[] inputs;
            public double output;
            public double[] weights;
            public activationFunction logFunc = activationFunction.sigmoid;
            public double[] dedw;
            public double delta = 0;
            //Momentum
            public double[] momentums;

            public void forward()
            {
                output = 0;
                for (int i = 0; i < inputs.Length; i++)
                {
                    output += inputs[i] * weights[i];
                }
                //output = math.logistics(output);
                output = math.logisticsFunc(logFunc, output);
            }

            public string toMath()
            {
                string ret = "";

                if (logFunc == activationFunction.relu)
                {
                    ret += "max(0, ";
                } else if (logFunc == activationFunction.sigmoid)
                {
                    ret += "sigmoid(";
                } else if (logFunc == activationFunction.tanh)
                {
                    ret += "tanh(";
                }

                for (int i = 0; i < weights.Length; i++)
                {
                    double weightValue = weights[i];
                    if (weightValue < 0) ret += "(" + "<pLayer[" + i + "]>" + "*(" + weightValue + "))";
                    else ret += "(" + "<pLayer[" + i + "]>" + "*" + weightValue + ")";
                    if (i < weights.Length - 1) ret += "+";
                }

                return ret + ")";
            }

            /*
            public neuron(int ni)
            {
				inputs = new double[ni];
				weights = new double[ni];
				for (int i = 0; i < ni; i++)
				{
					weights[i] = math.rand.NextDouble();
				}

				//Console.WriteLine("in");
			}
			*/

            public double getNetMomentum()
            {
                return momentums.Sum()/momentums.Length;
            }

            //Updated
            public void init(int ni, double[] initWeights)
            {
                //Console.WriteLine(ni);
				inputs = new double[ni];
				weights = new double[ni];
                dedw = new double[ni];
                momentums = new double[ni];
				for (int i = 0; i < ni; i++)
				{
                    weights[i] = initWeights[i];//0;//math.rand.NextDouble();
                    momentums[i] = 0;
				}

				//Console.WriteLine("in");
			}

			//TODO REPLACE output * (1 - output) with deriviative of logFunc
			//TODO IMPLEMENT WEIGHT MOMENTUM https://www.willamette.edu/~gorr/classes/cs449/momrate.html
			public void backwardOut(double expected)
            {
                //delta = output * (1 - output) * (output - expected);

                //Works!
                delta = math.derLogisticsFunc(logFunc,inputs.summation((x,y) => x*weights[y])) * (output - expected);
                //
                momentums = dedw;
                for (int i = 0; i < dedw.Length; i++)
                {
                    dedw[i] = delta * inputs[i];
                }
            }

            public void backwardHidden(layer myLayer, layer parentLayer)
            {
                double s = 0;
                int index = Array.IndexOf(myLayer.neurons, this);

                for (int i = 0; i < parentLayer.neurons.Length; i++)
                    s += parentLayer.neurons[i].weights[index] * parentLayer.neurons[i].delta;

                //delta = output * (1 - output) * s;//* (output - expected);
                delta = math.derLogisticsFunc(logFunc, inputs.summation((x, y) => x * weights[y])) * s;
				//math.derLogisticsFunc(logFunc, inputs.summation((x, y) => x * weights[y])) = output * (1 - output)

				for (int i = 0; i < dedw.Length; i++)
				{
					dedw[i] = delta * inputs[i];
				}
            }
            //TODO CHECK MOMENTUM FORMULA
            public void backwardWeights(double eta, double momentum)
            {
               // momentums.map((x,y)=>);
                for (int i = 0; i < weights.Length; i++)
                {
                    double wd = (eta * dedw[i]) + (momentums[i] * momentum);
                    //Console.WriteLine(weights[i]);
                    weights[i] -= wd;
                    //-WD OR +WD????
                    momentums[i] -= wd;
                }
                //momentums = weights;
            }
        }

        [Serializable]
        public class layer
        {
            public double[] outputs;
            public double[] inputs;
            public neuron[] neurons;
            public bool hasBias = false;
            public int ni;
            public Random random = new Random();
            //public string layerName;

            /*
            public layer(int nn, int ni)
            {
                neurons = new neuron[nn];
                outputs = new double[nn];
                for (int i = 0; i < nn; i++)
                {
                    neurons[i] = new neuron();
                    neurons[i].init(ni);
                }
            }
            */

            public void backwardOut(double[] expected)
            {
                for (int i = 0; i < neurons.Length; i++)
                {
                    neurons[i].backwardOut(expected[i]);
                }
            }

			public void backwardHidden(neuralNet nn)
			{
                int index = nn.layers.IndexOf(this);
				for (int i = 0; i < neurons.Length; i++)
				{
                    neurons[i].backwardHidden(this ,nn.layers[index+1]);
				}
                //Console.WriteLine(nn.layers[index + 1].layerName);
			}

            public void backwardWeights(double eta, double momentum)
            {
                for (int i = 0; i < neurons.Length; i++)
                {
                    neurons[i].backwardWeights(eta, momentum);
                }
                //Console.WriteLine("back");
            }

            //https://stats.stackexchange.com/questions/47590/what-are-good-initial-weights-in-a-neural-network/186351#186351
            /// <summary>
            /// Initializes the layer.
            /// </summary>
            /// <returns>The init.</returns>
            /// <param name="nn">Number of neurons</param>
            /// <param name="ni">Number of inputs per neuron</param>
            /// <param name="no">Number of outputs per neuron.</param>
            /// <param name="logFunc">Logistics function for the layer.</param>
            /// <param name="hb">If set to <c>true</c> will add bias to layer.</param>
            public void init(int nn, int ni, int no, activationFunction logFunc, bool hb)
            {
                if (hb) ni += 1;
                hasBias = hb;
				outputs = new double[nn];
                if (logFunc != activationFunction.input)
                {
                    neurons = new neuron[nn];
                    for (int i = 0; i < nn; i++)
                    {
                        neurons[i] = new neuron();
                        neurons[i].logFunc = logFunc;

                        double[] initWeights = new double[ni];
                        double r = 0;
                        if (logFunc.Equals(activationFunction.sigmoid))
                        {
                            r = 4 * Math.Sqrt(6/(ni + no));
                        }
                        else if (logFunc.Equals(activationFunction.tanh))
                        {
                            r = Math.Sqrt(6 / (ni + no));
                        }
                        else
                        {
                            r = 1;//IDK if this good
                        }
                        for (int j = 0; j < ni; j++)
                        {
                            initWeights[j] = random.NextDouble(-r, r);
                        }

                        neurons[i].init(ni, initWeights);
                    }
                }
            }

            public void forward()
            {
                //Console.WriteLine(hasBias);
                for (int i = 0; i < neurons.Length; i++)
                {
                    List<double> correctIn = inputs.ToList();
                    if (hasBias) correctIn.Add(1.0);
					neurons[i].inputs = correctIn.ToArray();
					neurons[i].forward();
                    outputs[i] = neurons[i].output;
                }
            }

            public string[] toMath()
            {
                List<string> neuronMath = new List<string>();

                foreach (neuron n in neurons)
                {
                    string neuronsMath = n.toMath();
                    if (hasBias)
                    {
                        //There will be an extra input for bias of 1
                        neuronsMath = neuronsMath.Replace("<pLayer[" + (n.inputs.Length - 1) + "]>*", "");
                    }
                    neuronMath.Add(neuronsMath);
                }

                return neuronMath.ToArray();
            }

            public double getNetMomentum()
            {
                double sum = 0;
                for (int i = 0; i < neurons.Length; i++) sum += neurons[i].getNetMomentum();
                return sum / neurons.Length;
            }
        }

        [Serializable]
        public class neuralNet
        {
            public List<layer> layers= new List<layer>();
            public double eta = 0;
            public double momentum = 0;

            public learningRateDescriptor learningDescriptor = new learningRateDescriptor();

            //public neuralNet(int numIn, int numHiddenL, int numHiddenNeurons, int NumOutputs)
            //{
            //    layers.Add(new layer(numIn,numIn));
            //}

            /// <summary>
            /// Forward the specified inputs.
            /// </summary>
            /// <returns>The output of the network</returns>
            /// <param name="inputs">Inputs for the network</param>
            public double[] forward(double[] inputs)
            {
                bool inputLayer = true;
                layer last = layers[0];
				layers[0].outputs = inputs;
				//layers[0].forward();
                foreach(layer l in layers)
                {
                    if(inputLayer)
                    {
                        inputLayer = false;
                        continue;
                    }
                    l.inputs = last.outputs;
                    l.forward();
                    last = l;
                }

                return layers[layers.Count-1].outputs;
            }

            public string toMath()
            {
                //string ret  = "";

                //1st layer is input
                //skip 2nd so first value is set
                string[] pLayer = layers[1].toMath();

                for (int i = 0; i < pLayer.Length; i++)
                {
                    for (int j = 0; j < layers[1].neurons.Length; j++)
                    {
                        pLayer[i] = pLayer[i].Replace("<pLayer[" + j + "]>", "in[" + j + "]");
                    }
                }

                for (int i = 2; i < layers.Count; i++)
                {
                    string[] cLayer = layers[i].toMath();
                    for (int j = 0; j < cLayer.Length; j++)
                    {
                        for (int k = 0; k < pLayer.Length; k++)
                        {
                            cLayer[j] = cLayer[j].Replace("<pLayer[" + k + "]>", pLayer[k]);
                        }
                    }
                    pLayer = cLayer;
                }

                return pLayer[0];
            }

            /// <summary>
            /// Backpropagate the specified Inputset and the expected values in Outputset.
            /// </summary>
            /// <param name="inputset">The set of inputs given.</param>
            /// <param name="outputset">The set of outputs expected for the inputs.</param>
            /// <param name="log">Log the new error.</param>
            public void backward(List<double[]> inputset, List<double[]> outputset, bool log)
            {
                backward(inputset, outputset);
                if(log) Console.WriteLine("New Error: " + getError(inputset, outputset));
            }

            /// <summary>
            /// Backpropagate the specified Inputset and the expected values in Outputset.
            /// </summary>
            /// <param name="inputset">The set of inputs given.</param>
            /// <param name="outputset">The set of outputs expected for the inputs.</param>
            public void backward(List<double[]> inputset, List<double[]> outputset)
            {
                for (int i = 0; i < inputset.Count; i++)
                {
                    backwardsPass(inputset[i], outputset[i]);
                }

                if (!learningDescriptor.learnMode.Equals(learningMode.staticLearn))
                {
                    eta = learningDescriptor.learningRate();
                }
            }

            /// <summary>
            /// A single pass using the backpropagation algorithm.
            /// </summary>
            /// <param name="inputset">A single frame of inputs.</param>
            /// <param name="outputset">A single frame of expected outputs.</param>
            public void backwardsPass(double[] inputset, double[] outputset)
            {
                forward(inputset);
				for (int i = layers.Count-1; i >= 0; i--)
				{
                    //Console.WriteLine(i);
					if (i == layers.Count-1)
					{
                        //outputLayer
                        layers[i].backwardOut(outputset);
                        continue;
					}
                    if(i == 0)
                    {
                        //InputLayer
                        for (int j = layers.Count - 1; j > 0; j--)
                            layers[j].backwardWeights(eta, momentum);
                        break;
                    }
                    //Hidden Layer
                    layers[i].backwardHidden(this);
				}
            }

            /// <summary>
            /// Calculates the error of the network given the input set: inputset and outputs for inputset: outputset.
            /// </summary>
            /// <returns>The square error.</returns>
            /// <param name="inputset">The set of input(s) to try.</param>
            /// <param name="outputset">The expected output(s).</param>
            public double getError(List<double[]> inputset, List<double[]> outputset)
            {
				double[] errors = new double[outputset[0].Length];
				for (int i = 0; i < inputset.Count; i++)
				{
					double[] outputs = forward(inputset[i]);
					for (int j = 0; j < outputset[i].Length; j++)
					{
						errors[j] += Math.Pow(0.5 * (outputs[j] - outputset[i][j]), 2);
                        //errors[j] += Math.Pow((outputs[j] - outputset[i][j]), 2);
					}
				}
                //return 0.5 * errors.Sum();
                return errors.Sum();
            }

            /// <summary>
            /// Saves the network specified path. NOTE: The save path is automatically appended with ".xml".
            /// </summary>
            /// <param name="path">The path to save the network to.</param>
            public void save(string path)
            {
                if (File.Exists(path + ".xml")) File.Delete(path + ".xml");
                FileStream fs = new FileStream(path+".xml", FileMode.OpenOrCreate);
                System.Xml.Serialization.XmlSerializer s = new System.Xml.Serialization.XmlSerializer(typeof(neuralNet));
                s.Serialize(fs, this);
            }

            /// <summary>
            /// Adds a layer to the neural network.
            /// </summary>
            /// <param name="NumberOfNeurons">The number of neurons for the layer.</param>
            /// <param name="NumberOfInputsPerNeuron">The number of inputs for each neuron in the layer.</param>
            /// <param name="ActivationFunction">The activation function for the layer (The squashing function).</param>
            /// <param name="bias">Whether to add a bais or not.</param>
            public void addLayer(int NumberOfNeurons, int NumberOfInputsPerNeuron, int NumberOfOutputsPerNeuron, activationFunction ActivationFunction, bool bias)
            {
                layer l = new layer();
                //l.init(x, y, logFunc);
                if (layers.Count == 0) l.init(NumberOfNeurons, NumberOfInputsPerNeuron, NumberOfOutputsPerNeuron, ActivationFunction, bias);
                else l.init(NumberOfNeurons, NumberOfInputsPerNeuron, NumberOfOutputsPerNeuron, ActivationFunction, bias);//true);
                layers.Add(l);
            }

            /// <summary>
            /// Adds a layer to the neural network. Bias set to false.
            /// </summary>
            /// <param name="NumberOfNeurons">The number of neurons for the layer.</param>
            /// <param name="NumberOfInputsPerNeuron">The number of inputs for each neuron in the layer.</param>
            /// <param name="ActivationFunction">The activation function for the layer (The squashing function).</param>
            public void addLayer(int NumberOfNeurons, int NumberOfInputsPerNeuron, int NumberOfOutputsPerNeuron, activationFunction ActivationFunction)
            {
                addLayer(NumberOfNeurons, NumberOfInputsPerNeuron, NumberOfOutputsPerNeuron, ActivationFunction, false);
            }

            /// <summary>
            /// Calculates the average momentum of all weights.
            /// </summary>
            /// <returns>The average of the momentum property of all weights.</returns>
            public double getNetMomentum()
            {
                double sum = 0;
                for (int i = 1; i < layers.Count; i++)
                    sum += layers[i].getNetMomentum();
                return sum / (layers.Count - 1);
            }

            /// <summary>
            /// Sets the hyper parameters to train the neural network. The network momentum coefficient is set to 0.
            /// </summary>
            /// <param name="LearnMode">The function to change the learning rate by.</param>
            /// <param name="StartLearningRate">The starting learning rate.</param>
            public void setLearn(learningMode LearnMode, double StartLearningRate)
            {
                learningDescriptor.startLearnRate = StartLearningRate;
                learningDescriptor.learnMode = LearnMode;
                eta = StartLearningRate;
            }

            /// <summary>
            /// Sets the hyper parameters to train the neural network.
            /// </summary>
            /// <param name="LearnMode">The function to change the learning rate by.</param>
            /// <param name="StartLearningRate">The starting learning rate.</param>
            /// <param name="Momentum">The momentum coefficient of the weights.</param>
            public void setLearn(learningMode LearnMode, double StartLearningRate, double Momentum)
            {
                learningDescriptor.startLearnRate = StartLearningRate;
                learningDescriptor.learnMode = LearnMode;
                eta = StartLearningRate;
                momentum = Momentum;
            }

            public double solve(double[] inputs, double[] outputs, double targetError, double minVal, double maxVal)
            {
                return solve(inputs, outputs, targetError, minVal, maxVal, -1);
            }

            public double solve(double[] inputs, double[] outputs, double targetError, double minVal, double maxVal, int initDirection)
            {
                //verify if we have enough data to solve it
                if (inputs.Length != layers[0].outputs.Length - 1) 
                {
                    throw new Exception("Not enough inputs");
                }

                if (outputs.Length != layers[layers.Count - 1].outputs.Length)
                {
                    throw new Exception("Not enough outputs");
                }

                activationFunction activation = layers[1].neurons[0].logFunc;
                //double maxVal = 0;
                //double minVal = 0;
                double currentVal = 0.5;
                //double prevError = 0;
                int direction = initDirection;

                double prevError = forward(inputs.Concat(new double[] { currentVal }).ToArray()).summation((x, j) => Math.Pow(0.5 * (x - outputs[j]), 2));

                while (true)
                {
                    double error = forward(inputs.Concat(new double[] { currentVal }).ToArray()).summation((x, j) => Math.Pow(0.5 * (x - outputs[j]), 2));
                    if (targetError > error)
                    {
                        return currentVal;
                    }
                    if (direction == 1)
                    {
                        if (currentVal.Equals(maxVal)) return currentVal;
                        if (prevError < error) direction = -1;
                        double nextVal = currentVal + (currentVal / 2);
                        if (nextVal > maxVal) nextVal = maxVal;
                        currentVal = nextVal;
                    }
                    if (direction == -1)
                    {
                        if (currentVal.Equals(minVal)) return currentVal;
                        if (prevError < error) direction = 1;
                        double nextVal = currentVal - (currentVal / 2);
                        if (nextVal > maxVal) nextVal = maxVal;
                        currentVal = nextVal;
                    }
                    prevError = error;
                    //Console.WriteLine("Error: {0}, Direction: {1}, Value: {2}", error, direction, currentVal);
                }
            }
        }

        public static class math
        {
            public static Random rand = new Random();

			//public static double logistics(double x)
			//{
			//    
			//}

			//public static double dervLogistics(double x)
			//{
			//    return Math.Exp(-x)/Math.Pow(1 + Math.Exp(-x),2);
			//}

			//http://kawahara.ca/what-is-the-derivative-of-relu/

            public static double logisticsFunc(activationFunction funcName, double x)
            {
                switch (funcName)
                {
                    case activationFunction.relu:
                        return Math.Max(0.0, x);
                    case activationFunction.sigmoid:
                        return 1 / (1 + Math.Exp(-x));
                    case activationFunction.tanh:
                        return Math.Tanh(x);
                    default: throw new Exception("No function with name" + funcName);
                }
            }

            public static double derLogisticsFunc(activationFunction funcName, double x)
            {
				switch (funcName)
				{
                    case activationFunction.relu:
                        if (x < 0) return 0;
                        if (x > 0) return 1;
                        return 0;
                    case activationFunction.sigmoid:
                        return Math.Exp(x) / Math.Pow(1 + Math.Exp(x), 2);
                        //return logisticsFunc(activationFunction.sigmoid, x) * (1 - logisticsFunc(activationFunction.sigmoid, x));
                    case activationFunction.tanh:
                        return 4 / Math.Pow((Math.Exp(-x) + Math.Exp(x)), 2);
					default: throw new Exception("No function with name" + funcName);
				}
            }

            public static double map(double x, double in_min, double in_max, double out_min, double out_max)
            {
                return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
            }
        }
    }

	public static class myNNExtensions
	{
		public static void ForEach<T>(this IEnumerable<T> source, Action<T> action)
		{
			source.ThrowIfNull("source");
			action.ThrowIfNull("action");
			foreach (T element in source)
			{
				action(element);
			}
		}

		public static void ThrowIfNull(this object obj, string objName)
		{
			if (obj == null)
				throw new Exception(string.Format("{0} is null.", objName));
		}

        public static double summation(this Array source, Func<double,int,double> action)
        {
			source.ThrowIfNull("source");
			action.ThrowIfNull("action");
            if (!(source.IsArrayOf(typeof(Double)))) throw new Exception("Needs to be double array");
            double[] a = (double[])source;
            double sum = 0;
            for (int i = 0; i < a.Length; i++)
            {
                sum += action(a[i],i);
            }
            return sum;
        }

        public static void map(this Array source, Func<object,int,object> action)
        {
            for (int i = 0; i < source.Length; i++)
                source.SetValue(action(source.GetValue(i),i),i);
        }

		public static bool IsArrayOf(this Array array, Type type)
		{
			return array.GetType().GetElementType().Equals(type);
		}

        //https://stats.stackexchange.com/questions/178626/how-to-normalize-data-between-1-and-1
        public static List<double> normalize(this List<double> values, double a, double b)
        {
            List<double> normalized = new List<double>();
            double min = values.Min();
            double max = values.Max();
            double dmaxmin = max - min;
            for (int i = 0; i < values.Count; i++)
            {
                double x = values[i];
                normalized.Add((b-a)*((x-min)/dmaxmin)+a);
            }
            return normalized;
        }

        public static double NextDouble(
        this Random random,
        double minValue,
        double maxValue)
        {
            return random.NextDouble() * (maxValue - minValue) + minValue;
        }

        //public static List<double> 
	}
}
