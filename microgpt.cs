// Program.cs
// Port of @karpathy microgpt.py gist to a single, dependency-free C# file.
// Optimized version with performance improvements.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net.Http;

public static class Program
{
    // -----------------------------
    // Minimal scalar autograd engine
    // -----------------------------
    public sealed class Value
    {
        public double Data;
        public double Grad;

        private readonly Value[] _children;
        private readonly double[] _localGrads; // d(this)/d(child_i)

        public Value(double data)
        {
            Data = data;
            Grad = 0.0;
            _children = Array.Empty<Value>();
            _localGrads = Array.Empty<double>();
        }

        private Value(double data, Value[] children, double[] localGrads)
        {
            Data = data;
            Grad = 0.0;
            _children = children;
            _localGrads = localGrads;
        }

        public static Value operator +(Value a, Value b)
            => new Value(a.Data + b.Data, new[] { a, b }, new[] { 1.0, 1.0 });

        public static Value operator +(Value a, double b) => a + new Value(b);
        public static Value operator +(double a, Value b) => new Value(a) + b;

        public static Value operator *(Value a, Value b)
            => new Value(a.Data * b.Data, new[] { a, b }, new[] { b.Data, a.Data });

        public static Value operator *(Value a, double b) => a * new Value(b);
        public static Value operator *(double a, Value b) => new Value(a) * b;

        public static Value operator -(Value a) => a * -1.0;
        public static Value operator -(Value a, Value b) => a + (-b);
        public static Value operator -(Value a, double b) => a + (-new Value(b));
        public static Value operator -(double a, Value b) => new Value(a) + (-b);

        public static Value operator /(Value a, Value b) => a * b.Pow(-1.0);
        public static Value operator /(Value a, double b) => a * new Value(b).Pow(-1.0);
        public static Value operator /(double a, Value b) => new Value(a) * b.Pow(-1.0);

        public Value Pow(double p)
        {
            // y = x^p ; dy/dx = p*x^(p-1)
            var y = Math.Pow(Data, p);
            var local = p * Math.Pow(Data, p - 1.0);
            return new Value(y, new[] { this }, new[] { local });
        }

        public Value Log()
        {
            // y = log(x) ; dy/dx = 1/x
            return new Value(Math.Log(Data), new[] { this }, new[] { 1.0 / Data });
        }

        public Value Exp()
        {
            // y = exp(x) ; dy/dx = exp(x)
            var e = Math.Exp(Data);
            return new Value(e, new[] { this }, new[] { e });
        }

        public Value Relu()
        {
            // y = max(0,x) ; dy/dx = 1 if x>0 else 0
            var y = Math.Max(0.0, Data);
            var local = Data > 0.0 ? 1.0 : 0.0;
            return new Value(y, new[] { this }, new[] { local });
        }

        public void Backward()
        {
            // topo sort
            var topo = new List<Value>(capacity: 1024);
            var visited = new HashSet<Value>(ReferenceEqualityComparer<Value>.Instance);

            void Build(Value v)
            {
                if (visited.Add(v))
                {
                    foreach (var ch in v._children) Build(ch);
                    topo.Add(v);
                }
            }

            Build(this);
            Grad = 1.0;

            for (int i = topo.Count - 1; i >= 0; i--)
            {
                var v = topo[i];
                for (int ci = 0; ci < v._children.Length; ci++)
                {
                    var child = v._children[ci];
                    var local = v._localGrads[ci];
                    child.Grad += local * v.Grad;
                }
            }
        }
    }

    // Reference equality comparer for topo traversal
    private sealed class ReferenceEqualityComparer<T> : IEqualityComparer<T> where T : class
    {
        public static readonly ReferenceEqualityComparer<T> Instance = new();
        public bool Equals(T? x, T? y) => ReferenceEquals(x, y);
        public int GetHashCode(T obj) => System.Runtime.CompilerServices.RuntimeHelpers.GetHashCode(obj);
    }

    // -----------------------------
    // Random helpers (Gaussian + weighted choice)
    // -----------------------------
    private static readonly Random Rng = new Random(42);

    private static double NextGaussian(double mean = 0.0, double std = 1.0)
    {
        // Box-Muller transform
        double u1 = 1.0 - Rng.NextDouble();
        double u2 = 1.0 - Rng.NextDouble();
        double z0 = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
        return mean + std * z0;
    }

    private static int WeightedChoice(IReadOnlyList<double> weights)
    {
        double total = 0.0;
        for (int i = 0; i < weights.Count; i++) total += weights[i];

        double r = Rng.NextDouble() * total;
        double c = 0.0;
        for (int i = 0; i < weights.Count; i++)
        {
            c += weights[i];
            if (r <= c) return i;
        }
        return weights.Count - 1; // fallback
    }

    // -----------------------------
    // Model hyperparams (same as python)
    // -----------------------------
    private const int n_embd = 16;
    private const int n_head = 4;
    private const int n_layer = 1;
    private const int block_size = 8;
    private const int head_dim = n_embd / n_head;

    // OPTIMIZATION: Cache sqrt(head_dim) calculation
    private static readonly double InvSqrtHeadDim = 1.0 / Math.Sqrt(head_dim);

    // state dict of parameter matrices
    private static Dictionary<string, List<List<Value>>> state_dict = new();
    private static List<Value> parameters = new();

    // OPTIMIZATION: Cache for character to ID mapping
    private static Dictionary<char, int> charToId = new();

    private static List<List<Value>> Matrix(int nout, int nin, double std = 0.02)
    {
        var mat = new List<List<Value>>(nout);
        for (int o = 0; o < nout; o++)
        {
            var row = new List<Value>(nin);
            for (int i = 0; i < nin; i++) row.Add(new Value(NextGaussian(0.0, std)));
            mat.Add(row);
        }
        return mat;
    }

    private static void InitModel(int vocabSize)
    {
        state_dict = new Dictionary<string, List<List<Value>>>();

        state_dict["wte"] = Matrix(vocabSize, n_embd);
        state_dict["wpe"] = Matrix(block_size, n_embd);
        state_dict["lm_head"] = Matrix(vocabSize, n_embd);

        for (int i = 0; i < n_layer; i++)
        {
            state_dict[$"layer{i}.attn_wq"] = Matrix(n_embd, n_embd);
            state_dict[$"layer{i}.attn_wk"] = Matrix(n_embd, n_embd);
            state_dict[$"layer{i}.attn_wv"] = Matrix(n_embd, n_embd);
            state_dict[$"layer{i}.attn_wo"] = Matrix(n_embd, n_embd, std: 0.0);
            state_dict[$"layer{i}.mlp_fc1"] = Matrix(4 * n_embd, n_embd);
            state_dict[$"layer{i}.mlp_fc2"] = Matrix(n_embd, 4 * n_embd, std: 0.0);
        }

        parameters = new List<Value>();
        foreach (var mat in state_dict.Values)
            foreach (var row in mat)
                parameters.AddRange(row);
    }

    // -----------------------------
    // Model ops (linear, softmax, rmsnorm, gpt)
    // -----------------------------
    private static List<Value> Linear(List<Value> x, List<List<Value>> w)
    {
        // w: [nout][nin], x: [nin] -> out: [nout]
        var output = new List<Value>(w.Count);
        for (int o = 0; o < w.Count; o++)
        {
            Value sum = new Value(0.0);
            var row = w[o];
            for (int i = 0; i < row.Count; i++)
                sum = sum + (row[i] * x[i]);
            output.Add(sum);
        }
        return output;
    }

    private static List<Value> Softmax(List<Value> logits)
    {
        // OPTIMIZATION: Manual max instead of LINQ
        double maxVal = double.NegativeInfinity;
        for (int i = 0; i < logits.Count; i++)
        {
            if (logits[i].Data > maxVal)
                maxVal = logits[i].Data;
        }

        var exts = new List<Value>(logits.Count);
        for (int i = 0; i < logits.Count; i++)
            exts.Add((logits[i] - maxVal).Exp());

        Value total = new Value(0.0);
        for (int i = 0; i < exts.Count; i++) total = total + exts[i];

        var probs = new List<Value>(logits.Count);
        for (int i = 0; i < exts.Count; i++) probs.Add(exts[i] / total);
        return probs;
    }

    private static List<Value> RmsNorm(List<Value> x)
    {
        // ms = mean(x^2), scale = (ms + 1e-5)^-0.5, return x*scale
        Value ms = new Value(0.0);
        for (int i = 0; i < x.Count; i++) ms = ms + (x[i] * x[i]);
        ms = ms / (double)x.Count;

        Value scale = (ms + 1e-5).Pow(-0.5);

        var y = new List<Value>(x.Count);
        for (int i = 0; i < x.Count; i++) y.Add(x[i] * scale);
        return y;
    }

    private static List<Value> Slice(List<Value> v, int start, int length)
    {
        var r = new List<Value>(length);
        for (int i = 0; i < length; i++) r.Add(v[start + i]);
        return r;
    }

    private static List<Value> GPT(
        int tokenId,
        int posId,
        List<List<List<Value>>> keys,    // [layer][time][n_embd]
        List<List<List<Value>>> values)  // [layer][time][n_embd]
    {
        var tok_emb = state_dict["wte"][tokenId];   // length n_embd
        var pos_emb = state_dict["wpe"][posId];     // length n_embd

        var x = new List<Value>(n_embd);
        for (int i = 0; i < n_embd; i++) x.Add(tok_emb[i] + pos_emb[i]);

        x = RmsNorm(x);

        for (int li = 0; li < n_layer; li++)
        {
            // 1) Multi-head attention block
            var x_residual = x;
            x = RmsNorm(x);

            var q = Linear(x, state_dict[$"layer{li}.attn_wq"]);
            var k = Linear(x, state_dict[$"layer{li}.attn_wk"]);
            var v = Linear(x, state_dict[$"layer{li}.attn_wv"]);

            keys[li].Add(k);
            values[li].Add(v);

            var x_attn = new List<Value>(n_embd);

            for (int h = 0; h < n_head; h++)
            {
                int hs = h * head_dim;

                var q_h = Slice(q, hs, head_dim);

                // OPTIMIZATION: Avoid ToList() - slice on demand
                int timeSteps = keys[li].Count;
                var attn_logits = new List<Value>(timeSteps);
                
                for (int t = 0; t < timeSteps; t++)
                {
                    Value dot = new Value(0.0);
                    var k_t = keys[li][t];
                    
                    for (int j = 0; j < head_dim; j++)
                        dot = dot + (q_h[j] * k_t[hs + j]);

                    // OPTIMIZATION: Use cached inverse sqrt
                    attn_logits.Add(dot * InvSqrtHeadDim);
                }

                var attn_weights = Softmax(attn_logits);
                
                // head_out[j] = sum_t attn_weights[t] * v_h[t][j]
                for (int j = 0; j < head_dim; j++)
                {
                    Value sum = new Value(0.0);
                    for (int t = 0; t < timeSteps; t++)
                    {
                        var v_t = values[li][t];
                        sum = sum + (attn_weights[t] * v_t[hs + j]);
                    }
                    x_attn.Add(sum);
                }
            }

            x = Linear(x_attn, state_dict[$"layer{li}.attn_wo"]);
            var x_added = new List<Value>(n_embd);
            for (int i = 0; i < n_embd; i++) x_added.Add(x[i] + x_residual[i]);
            x = x_added;

            // 2) MLP block
            x_residual = x;
            x = RmsNorm(x);
            x = Linear(x, state_dict[$"layer{li}.mlp_fc1"]);
            for (int i = 0; i < x.Count; i++) x[i] = x[i].Relu().Pow(2.0);
            x = Linear(x, state_dict[$"layer{li}.mlp_fc2"]);

            x_added = new List<Value>(n_embd);
            for (int i = 0; i < n_embd; i++) x_added.Add(x[i] + x_residual[i]);
            x = x_added;
        }

        var logits = Linear(x, state_dict["lm_head"]);
        return logits;
    }

    // -----------------------------
    // Main: dataset, tokenizer, train, inference
    // -----------------------------
    public static void Main()
    {
        // Dataset
        const string inputPath = "input.txt";
        if (!File.Exists(inputPath))
        {
            Console.WriteLine("input.txt not found, downloading names dataset...");
            var url = "https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt";
            using var http = new HttpClient();
            var bytes = http.GetByteArrayAsync(url).GetAwaiter().GetResult();
            File.WriteAllBytes(inputPath, bytes);
        }

        var docs = File.ReadAllLines(inputPath)
            .Select(l => l.Trim())
            .Where(l => !string.IsNullOrWhiteSpace(l))
            .ToList();

        // Shuffle
        docs = docs.OrderBy(_ => Rng.Next()).ToList();
        Console.WriteLine($"num docs: {docs.Count}");

        // Tokenizer: unique chars become 0..n-1, BOS is last id
        var uchars = docs.SelectMany(s => s).Distinct().OrderBy(c => c).ToList();
        int BOS = uchars.Count;
        int vocabSize = uchars.Count + 1;

        // OPTIMIZATION: Build char->id dictionary for O(1) lookup
        charToId = new Dictionary<char, int>(uchars.Count);
        for (int i = 0; i < uchars.Count; i++)
        {
            charToId[uchars[i]] = i;
        }

        Console.WriteLine($"vocab size: {vocabSize}");

        // Model init
        InitModel(vocabSize);
        Console.WriteLine($"num params: {parameters.Count}");

        // Adam buffers
        double learningRate = 1e-2, beta1 = 0.9, beta2 = 0.95, epsAdam = 1e-8;
        var m = new double[parameters.Count];
        var v = new double[parameters.Count];

        // OPTIMIZATION: Pre-calculate beta powers for Adam
        var beta1Powers = new double[500];
        var beta2Powers = new double[500];
        for (int i = 0; i < 500; i++)
        {
            beta1Powers[i] = Math.Pow(beta1, i + 1);
            beta2Powers[i] = Math.Pow(beta2, i + 1);
        }

        // Train
        int numSteps = 500;
        for (int step = 0; step < numSteps; step++)
        {
            string doc = docs[step % docs.Count];

            // tokens = [BOS] + chars + [BOS]
            var tokens = new List<int>(doc.Length + 2) { BOS };
            
            // OPTIMIZATION: Use dictionary lookup instead of IndexOf
            for (int i = 0; i < doc.Length; i++)
                tokens.Add(charToId[doc[i]]);
            tokens.Add(BOS);

            int n = Math.Min(block_size, tokens.Count - 1);

            var keys = Enumerable.Range(0, n_layer).Select(_ => new List<List<Value>>()).ToList();
            var values = Enumerable.Range(0, n_layer).Select(_ => new List<List<Value>>()).ToList();

            var losses = new List<Value>(n);

            for (int posId = 0; posId < n; posId++)
            {
                int tokenId = tokens[posId];
                int targetId = tokens[posId + 1];

                var logits = GPT(tokenId, posId, keys, values);
                var probs = Softmax(logits);

                var loss_t = -probs[targetId].Log();
                losses.Add(loss_t);
            }

            // loss = mean(losses)
            Value loss = new Value(0.0);
            for (int i = 0; i < losses.Count; i++) loss = loss + losses[i];
            loss = (1.0 / n) * loss;

            // backward
            loss.Backward();

            // cosine lr decay
            double lr_t = learningRate * 0.5 * (1.0 + Math.Cos(Math.PI * step / numSteps));

            // adam update - OPTIMIZATION: Use pre-calculated powers
            double beta1Factor = 1.0 - beta1Powers[step];
            double beta2Factor = 1.0 - beta2Powers[step];

            for (int i = 0; i < parameters.Count; i++)
            {
                var p = parameters[i];
                m[i] = beta1 * m[i] + (1.0 - beta1) * p.Grad;
                v[i] = beta2 * v[i] + (1.0 - beta2) * (p.Grad * p.Grad);

                double mHat = m[i] / beta1Factor;
                double vHat = v[i] / beta2Factor;

                p.Data -= lr_t * mHat / (Math.Sqrt(vHat) + epsAdam);
                p.Grad = 0.0;
            }

            Console.WriteLine($"step {step + 1,4} / {numSteps,4} | loss {loss.Data:F4}");
        }

        // Inference
        double temperature = 0.5;
        Console.WriteLine("\n--- inference ---");

        for (int sampleIdx = 0; sampleIdx < 20; sampleIdx++)
        {
            var keys = Enumerable.Range(0, n_layer).Select(_ => new List<List<Value>>()).ToList();
            var values = Enumerable.Range(0, n_layer).Select(_ => new List<List<Value>>()).ToList();

            int tokenId = BOS;
            var sample = new List<char>();

            for (int posId = 0; posId < block_size; posId++)
            {
                var logits = GPT(tokenId, posId, keys, values);

                // probs = softmax(logits / temperature)
                var scaled = new List<Value>(logits.Count);
                for (int i = 0; i < logits.Count; i++) scaled.Add(logits[i] / temperature);

                var probs = Softmax(scaled);
                var w = probs.Select(p => p.Data).ToList();

                tokenId = WeightedChoice(w);

                if (tokenId == BOS) break;
                sample.Add(uchars[tokenId]);
            }

            Console.WriteLine($"sample {sampleIdx + 1,2}: {new string(sample.ToArray())}");
        }
    }
}
