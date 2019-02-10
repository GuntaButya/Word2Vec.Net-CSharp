namespace Word2Vec
{



    #region Using Statements:



    using System;
    using System.IO;
    using System.Threading;
    using System.Diagnostics;
    using System.Threading.Tasks;
    using System.Text.RegularExpressions;
    using Utils;



    #endregion



    /// <summary>
    /// Word2Vec - By Tomas Mikolov
    /// This tool provides an efficient implementation of the continuous bag-of-words and skip-gram architectures for computing vector representations of words. 
    /// These representations can be subsequently used in many natural language processing applications and for further research.
    /// Project: https://code.google.com/archive/p/word2vec/#
    /// Citations: https://scholar.google.com/citations?user=oBu8kMMAAAAJ
    /// Questions: https://groups.google.com/forum/#!forum/word2vec-toolkit
    /// Source Code: https://code.google.com/archive/p/word2vec/source/default/source
    /// Paper: https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf
    /// </summary>
    public class Word2Vec
    {



        #region Fields:



        private const int MaxExp = 6;
        private const int MaxCodeLength = 40;
        private const int ExpTableSize = 1000;
        private const int VocabHashSize = 30000000;
        private const int MaxSentenceLength = 1000;



        private readonly string _trainFile;
        private readonly string _outputFile;
        private readonly string _saveVocabFile;
        private readonly string _readVocabFile;



        // Defaults:
        private int _binary = 0;
        private int _cbow = 1;
        private int _debugMode = 2;
        private int _minCount = 5;
        private int _numThreads = 12;
        private int _layer1Size = 100;
        private long _iter = 5;
        private long _classes = 0;
        private float _alpha = (float)0.025;
        private float _sample = (float)1e-3;

        private int _hs = 0;
        private int _negative = 5;
        private int _window = 5;



        private float[] _syn0;
        private float[] _syn1;
        private float[] _syn1Neg;

        private VocabWord[] _vocab;

        private int _minReduce = 1;

        private readonly int[] _vocabHash;

        private int _vocabMaxSize = 1000;

        private int _vocabSize;

        private long _trainWords;

        private long _wordCountActual;

        private long _fileSize;

        private float _startingAlpha;

        private readonly float[] _expTable;

        private Stopwatch _stopwatch;

        private const int TableSize = (int)1e8;

        private int[] _table;



        #endregion



        #region Properties:

        #endregion




        /// <summary>
        /// Speciffically for Training on a Text Corpus of large quantity. The bigger the better generally.
        /// Example Usage:
        /// From the C++ Class: './word2vec -train data.txt -output vec.txt -size 200 -window 5 -sample 1e-4 -negative 5 -hs 0 -binary 0 -cbow 1 -iter 3'
        /// demo-analogy: './word2vec -train text8 -output vectors.bin -cbow 1 -size 200 -window 8 -negative 25 -hs 0 -sample 1e-4 -threads 20 -binary 1 -iter 15'
        /// demo-classes: './word2vec -train text8 -output classes.txt -cbow 1 -size 200 -window 8 -negative 25 -hs 0 -sample 1e-4 -threads 20 -iter 15 -classes 500 (sort classes.txt -k 2 -n > classes.sorted.txt) - not sure on this bit?'
        /// demo-word: './word2vec -train text8 -output vectors.bin -cbow 1 -size 200 -window 8 -negative 25 -hs 0 -sample 1e-4 -threads 20 -binary 1 -iter 15'
        /// demo-word-accuracy: './word2vec -train text8 -output vectors.bin -cbow 1 -size 200 -window 8 -negative 25 -hs 0 -sample 1e-4 -threads 20 -binary 1 -iter 15'
        /// </summary>
        /// <param name="train">-train <file> Use text data from <file> to train the model</param>
        /// <param name="output">-output <file> Use <file> to save the resulting word vectors / word clusters</param>
        /// <param name="save-vocab">-save-vocab <file> The vocabulary will be saved to <file></param>
        /// <param name="readvocab">-read-vocab <file> The vocabulary will be read from <file>, not constructed from the training data</param>
        /// <param name="size">-size <int> Set size of word vectors; default is 100</param>
        /// <param name="debug">-debug <int> Set the debug mode (default = 2 = more info during training)</param>
        /// <param name="binary">-binary <int> Save the resulting vectors in binary moded; default is 0 (off)</param>
        /// <param name="cbow">-cbow <int> Use the continuous bag of words model; default is 1 (use 0 for skip-gram model)</param>
        /// <param name="alpha">-alpha <float> Set the starting learning rate; default is 0.025 for skip-gram and 0.05 for CBOW</param>
        /// <param name="sample">-sample <float> Set threshold for occurrence of words. Those that appear with higher frequency in the training data</param>
        /// <param name="hs">-hs <int> Use Hierarchical Softmax; default is 0 (not used)</param>
        /// <param name="negative">-negative <int> Number of negative examples; default is 5, common values are 3 - 10 (0 = not used)</param>
        /// <param name="threads">-threads <int> Use <int> threads (default 12)</param>
        /// <param name="iter">-iter <int> Run more training iterations (default 5)</param>
        /// <param name="mincount">-min-count <int> This will discard words that appear less than <int> times; default is 5</param>
        /// <param name="classes">-classes <int> Output word classes rather than word vectors; default number of classes is 0 (vectors are written)</param>
        /// <param name="window">-window <int> Set max skip length between words; default is 5</param>
        public Word2Vec(string train, string output, string savevocab, string readvocab, int size, int debug, int binary, int cbow, float alpha, float sample, int hs, int negative, int threads, long iter, int mincount, long classes, int window)
        {

            _trainFile = train;
            _outputFile = output;
            _saveVocabFile = savevocab;
            _readVocabFile = readvocab;
            _vocab = new VocabWord[_vocabMaxSize];
            _vocabHash = new int[VocabHashSize];
            _expTable = new float[ExpTableSize + 1];

            for (var i = 0; i < ExpTableSize; i++)
            {

                // Precompute the exp() table:
                // Bug Fix see: https://github.com/GuntaButya/Word2Vec.Net-CSharp/issues/1
                _expTable[i] = (float)Math.Exp((((double)i) / ((double)ExpTableSize) * 2.0 - 1.0) * ((double)MaxExp));

                // Precompute f(x) = x / (x + 1):
                _expTable[i] = _expTable[i] / (_expTable[i] + 1);
            }

            
            _layer1Size = size;
            _debugMode = debug;
            _binary = binary;
            _cbow = cbow;
            _alpha = alpha;
            _sample = sample;
            _hs = hs;
            _negative = negative;
            _numThreads = threads;
            _iter = iter;
            _minCount = mincount;
            _classes = classes;
            _window = window;
        }



        /// <summary>
        /// Initialises a Unigram Table.
        /// </summary>
        private void InitUnigramTable()
        {

            int a;

            double trainWordsPow = 0;

            double power = 0.75;

            _table = new int[TableSize];

            for (a = 0; a < _vocabSize; a++)
                trainWordsPow += Math.Pow(_vocab[a].Cn, power);

            int i = 0;

            var d1 = Math.Pow(_vocab[i].Cn, power) / trainWordsPow;

            for (a = 0; a < TableSize; a++)
            {
                _table[a] = i;

                if (a / (double)TableSize > d1)
                {
                    i++;

                    d1 += Math.Pow(_vocab[i].Cn, power) / trainWordsPow;
                }

                if (i >= _vocabSize)
                    i = _vocabSize - 1;

            }

        }



        /// <summary>
        /// Gets a Hash for the Word.
        /// </summary>
        private uint GetWordHash(string word)
        {
            int a;

            ulong hash = 0;

            for (a = 0; a < word.Length; a++)
                hash = hash * 257 + word[a];

            hash = hash % VocabHashSize;

            return (uint)hash;
        }



        /// <summary>
        /// Searching the word position in the vocabulary
        /// </summary>
        /// <param name="word"></param>
        /// <returns>position of a word in the vocabulary; if the word is not found, returns -1</returns>
        private int SearchVocab(string word)
        {

            var hash = GetWordHash(word);

            while (true)
            {
                if (_vocabHash[hash] == -1)
                    return -1;

                if (word.Equals(_vocab[_vocabHash[hash]].Word))
                    return _vocabHash[hash];

                hash = (hash + 1) % VocabHashSize;
            }

            return -1;
        }



        /// <summary>
        /// Adds a word to the vocabulary
        /// </summary>
        /// <param name="word"></param>
        /// <returns></returns>
        private int AddWordToVocab(string word)
        {

            _vocab[_vocabSize].Word = word;

            _vocab[_vocabSize].Cn = 0;

            _vocabSize++;

            // Resize array if needed
            if (_vocabSize + 2 >= _vocabMaxSize)
            {
                _vocabMaxSize += 1000;
                Array.Resize(ref _vocab, _vocabMaxSize);
            }

            uint hash = GetWordHash(word);

            while (_vocabHash[hash] != -1)
                hash = (hash + 1) % VocabHashSize;

            _vocabHash[hash] = _vocabSize - 1;

            return _vocabSize - 1;
        }



        /// <summary>
        /// Sorts the vocabulary by frequency using word counts
        /// </summary>
        private void SortVocab()
        {

            // Sort the vocabulary and keep </s> at the first position:
            Array.Sort(_vocab, 1, _vocabSize - 1, new VocubComparer());

            for (var a = 0; a < VocabHashSize; a++)
                _vocabHash[a] = -1;

            int size = _vocabSize;

            _trainWords = 0;

            for (var a = 0; a < size; a++)
            {
                // Words occuring less than min_count times will be discarded from the vocab:
                if (_vocab[a].Cn < _minCount && (a != 0))
                {
                    _vocabSize--;

                    _vocab[a].Word = null;
                }
                else
                {
                    // Hash will be re-computed, as after the sorting it is not actual:
                    var hash = GetWordHash(_vocab[a].Word);

                    while (_vocabHash[hash] != -1)
                        hash = (hash + 1) % VocabHashSize;
                    _vocabHash[hash] = a;

                    _trainWords += _vocab[a].Cn;

                }

            }

            Array.Resize(ref _vocab, _vocabSize + 1);

            // Allocate memory for the binary tree construction:
            for (var a = 0; a < _vocabSize; a++)
            {
                _vocab[a].Code = new char[MaxCodeLength];

                _vocab[a].Point = new int[MaxCodeLength];
            }

            GC.Collect();

        }



        /// <summary>
        /// Reduces the vocabulary by removing infrequent tokens
        /// </summary>
        private void ReduceVocab()
        {

            var b = 0;

            for (var a = 0; a < _vocabSize; a++)
            {
                if (_vocab[a].Cn > _minReduce)
                {
                    _vocab[b].Cn = _vocab[a].Cn;

                    _vocab[b].Word = _vocab[a].Word;

                    b++;
                }
                else
                    _vocab[a].Word = null;

            }

            _vocabSize = b;

            for (var a = 0; a < VocabHashSize; a++)
                _vocabHash[a] = -1;

            for (var a = 0; a < _vocabSize; a++)
            {

                // Hash will be re-computed, as it is not actual
                uint hash = GetWordHash(_vocab[a].Word);

                while (_vocabHash[hash] != -1)
                    hash = (hash + 1) % VocabHashSize;

                _vocabHash[hash] = a;

            }

            _minReduce++;

            GC.Collect();
        }



        /// <summary>
        /// Create binary Huffman tree using the word counts
        /// Frequent words will have short uniqe binary codes
        /// </summary>
        private void CreateBinaryTree()
        {

            long pos1;

            long pos2;

            var code = new char[MaxCodeLength];

            var point = new long[MaxCodeLength];

            var count = new long[_vocabSize * 2 + 1];

            var binary = new long[_vocabSize * 2 + 1];

            var parentNode = new int[_vocabSize * 2 + 1];

            for (var a = 0; a < _vocabSize; a++)
                count[a] = _vocab[a].Cn;

            for (var a = _vocabSize; a < _vocabSize * 2; a++)
                count[a] = (long)1e15;

            pos1 = _vocabSize - 1;

            pos2 = _vocabSize;

            // Following algorithm constructs the Huffman tree by adding one node at a time
            for (var a = 0; a < _vocabSize - 1; a++)
            {
                // First, find two smallest nodes 'min1, min2'
                long min1i;

                if (pos1 >= 0)
                {
                    if (count[pos1] < count[pos2])
                    {
                        min1i = pos1;
                        pos1--;
                    }
                    else
                    {
                        min1i = pos2;
                        pos2++;
                    }
                }
                else
                {
                    min1i = pos2;
                    pos2++;
                }

                long min2i;

                if (pos1 >= 0)
                {
                    if (count[pos1] < count[pos2])
                    {
                        min2i = pos1;
                        pos1--;
                    }
                    else
                    {
                        min2i = pos2;
                        pos2++;
                    }
                }
                else
                {
                    min2i = pos2;
                    pos2++;
                }

                count[_vocabSize + a] = count[min1i] + count[min2i];

                parentNode[min1i] = _vocabSize + a;

                parentNode[min2i] = _vocabSize + a;

                binary[min2i] = 1;
            }

            // Now assign binary code to each vocabulary word
            for (long a = 0; a < _vocabSize; a++)
            {

                var b = a;

                long i = 0;

                while (true)
                {
                    code[i] = (char)binary[b];

                    point[i] = b;

                    i++;

                    b = parentNode[b];

                    if (b == _vocabSize * 2 - 2)
                        break;

                }

                _vocab[a].CodeLen = (int)i;

                _vocab[a].Point[0] = _vocabSize - 2;

                for (b = 0; b < i; b++)
                {
                    _vocab[a].Code[i - b - 1] = code[b];

                    _vocab[a].Point[i - b] = (int)(point[b] - _vocabSize);
                }

            }

            GC.Collect();

        }



        /// <summary>
        /// Learn the Vocabulary from a Training File.
        /// </summary>
        private void LearnVocabFromTrainFile()
        {

            int i;

            for (var a = 0; a < VocabHashSize; a++)
                _vocabHash[a] = -1;

            using (var fin = File.OpenText(_trainFile))
            {

                if (fin == StreamReader.Null)
                {
                    throw new InvalidOperationException("ERROR: training data file not found!\n");
                }

                _vocabSize = 0;

                string line;

                Regex regex = new Regex("\\s");

                AddWordToVocab("</s>");

                while ((line = fin.ReadLine()) != null)
                {
                    if (fin.EndOfStream) break;

                    string[] words = regex.Split(line);

                    foreach (var word in words)
                    {
                        // string cleanedWord = word.Clean();
                        if (string.IsNullOrWhiteSpace(word))
                            continue;

                        _trainWords++;

                        if ((_debugMode > 1) && (_trainWords % 100000 == 0))
                        {
                            Console.Write("{0}K \r", _trainWords / 1000);
                        }

                        i = SearchVocab(word);

                        if (i == -1)
                        {
                            var a = AddWordToVocab(word);
                            _vocab[a].Cn = 1;
                        }
                        else
                            _vocab[i].Cn++;

                        if (_vocabSize > VocabHashSize * 0.7)
                            ReduceVocab();
                    }

                }

                SortVocab();

                if (_debugMode > 0)
                {
                    Console.WriteLine("Vocab size: {0}", _vocabSize);
                    Console.WriteLine("Words in train file: {0}", _trainWords);
                }

                _fileSize = new FileInfo(_trainFile).Length;
            }

        }



        /// <summary>
        /// Save the Vocabulary.
        /// </summary>
        private void SaveVocab()
        {
            using (var stream = new FileStream(_saveVocabFile, FileMode.OpenOrCreate))
            {
                using (var streamWriter = new StreamWriter(stream))
                {
                    for (var i = 0; i < _vocabSize; i++)
                        streamWriter.WriteLine("{0} {1}", _vocab[i].Word, _vocab[i].Cn);
                }
            }
        }



        /// <summary>
        /// Read the Vocabulary.
        /// </summary>
        private void ReadVocab()
        {

            for (var a = 0; a < VocabHashSize; a++)
                _vocabHash[a] = -1;

            _vocabSize = 0;

            using (var fin = File.OpenText(_readVocabFile))
            {
                string line;

                var regex = new Regex("\\s");

                while ((line = fin.ReadLine()) != null)
                {
                    var vals = regex.Split(line);

                    if (vals.Length == 2)
                    {
                        var a = AddWordToVocab(vals[0]);

                        _vocab[a].Cn = int.Parse(vals[1]);
                    }

                }

                SortVocab();

                if (_debugMode > 0)
                {
                    Console.WriteLine("Vocab size: {0}", _vocabSize);
                    Console.WriteLine("Words in train file: {0}", _trainWords);
                }

            }

            var fileInfo = new FileInfo(_trainFile);

            _fileSize = fileInfo.Length;
        }



        /// <summary>
        /// Initialises the Net.
        /// </summary>
        private void InitNet()
        {

            long a, b;

            ulong nextRandom = 1;

            _syn0 = new float[_vocabSize * _layer1Size];

            if (_hs > 0)
            {
                _syn1 = new float[_vocabSize * _layer1Size];

                for (a = 0; a < _vocabSize; a++)
                    for (b = 0; b < _layer1Size; b++)
                        _syn1[a * _layer1Size + b] = 0;
            }

            if (_negative > 0)
            {
                _syn1Neg = new float[_vocabSize * _layer1Size];

                for (a = 0; a < _vocabSize; a++)
                    for (b = 0; b < _layer1Size; b++)
                        _syn1Neg[a * _layer1Size + b] = 0;
            }

            for (a = 0; a < _vocabSize; a++)
            {
                for (b = 0; b < _layer1Size; b++)
                {
                    nextRandom = nextRandom * 25214903917 + 11;
                    _syn0[a * _layer1Size + b] = ((nextRandom & 0xFFFF) / (float)65536 - (float)0.5) / _layer1Size;
                }
            }

            CreateBinaryTree();

            GC.Collect();
        }



        /// <summary>
        /// Parallel Thread for Increased Training Speed.
        /// </summary>
        /// <param name="id">Thread Id</param>
        private void TrainModelThread(int id)
        {

            Regex splitRegex = new Regex("\\s");

            long sentenceLength = 0;

            long sentencePosition = 0;

            long wordCount = 0, lastWordCount = 0;

            var sen = new long[MaxSentenceLength + 1];

            long localIter = _iter;

            // Console.WriteLine("{0} started", id);

            Thread.Sleep(100);

            var nextRandom = (ulong)id;

            float g;

            var neu1 = new float[_layer1Size];

            var neu1e = new float[_layer1Size];

            using (StreamReader fi = File.OpenText(_trainFile))
            {

                fi.BaseStream.Seek(_fileSize / _numThreads * id, SeekOrigin.Begin);

                while (true)
                {
                    if (wordCount - lastWordCount > 10000)
                    {
                        _wordCountActual += wordCount - lastWordCount;

                        lastWordCount = wordCount;

                        if (_debugMode > 1)
                        {
                            Console.Write("\rAlpha: {0}  Progress: {1:0.00}%  Words/thread/sec:{2:0.00}K ", _alpha, _wordCountActual / (float)(_iter * _trainWords + 1) * 100, _wordCountActual / (float)_stopwatch.ElapsedMilliseconds);//*1000
                        }

                        _alpha = _startingAlpha * (1 - _wordCountActual / (float)(_iter * _trainWords + 1));

                        if (_alpha < _startingAlpha * (float)0.0001)
                            _alpha = _startingAlpha * (float)0.0001;

                    }

                    long word;

                    if (sentenceLength == 0)
                    {
                        string line;

                        bool loopEnd = false;

                        while (!loopEnd && (line = fi.ReadLine()) != null)
                        {
                            string[] words = splitRegex.Split(line);

                            foreach (var s in words)
                            {
                                word = SearchVocab(s);

                                if (fi.EndOfStream)
                                {
                                    loopEnd = true;
                                    break;
                                }

                                if (word == -1)
                                    continue;

                                wordCount++;

                                if (word == 0)
                                {
                                    loopEnd = true;
                                    break;
                                }

                                // The subsampling randomly discards frequent words while keeping the ranking same
                                if (_sample > 0)
                                {
                                    var ran = ((float)Math.Sqrt(_vocab[word].Cn / (_sample * _trainWords)) + 1) * (_sample * _trainWords) / _vocab[word].Cn;

                                    nextRandom = nextRandom * 25214903917 + 11;

                                    if (ran < (nextRandom & 0xFFFF) / (float)65536)
                                        continue;
                                }

                                sen[sentenceLength] = word;

                                sentenceLength++;

                                if (sentenceLength >= MaxSentenceLength)
                                {
                                    loopEnd = true;
                                    break;
                                }

                            }

                        }

                        sentencePosition = 0;
                    }

                    if (fi.EndOfStream || (wordCount > _trainWords / _numThreads))
                    {
                        _wordCountActual += wordCount - lastWordCount;

                        localIter--;

                        if (localIter == 0)
                            break;

                        wordCount = 0;

                        lastWordCount = 0;

                        sentenceLength = 0;

                        fi.BaseStream.Seek(_fileSize / _numThreads * id, SeekOrigin.Begin);

                        continue;
                    }

                    word = sen[sentencePosition];

                    if (word == -1)
                        continue;

                    long c;

                    for (c = 0; c < _layer1Size; c++)
                        neu1[c] = 0;

                    for (c = 0; c < _layer1Size; c++)
                        neu1e[c] = 0;

                    nextRandom = nextRandom * 25214903917 + 11;

                    var b = (long)(nextRandom % (ulong)_window);

                    long label;

                    long lastWord;

                    long d;

                    float f;

                    long target;

                    long l2;

                    if (_cbow > 0)
                    {
                        //train the cbow architecture
                        // in -> hidden
                        long cw = 0;

                        for (var a = b; a < _window * 2 + 1 - b; a++)
                            if (a != _window)
                            {
                                c = sentencePosition - _window + a;

                                if (c < 0)
                                    continue;

                                if (c >= sentenceLength)
                                    continue;

                                lastWord = sen[c];

                                if (lastWord == -1)
                                    continue;

                                for (c = 0; c < _layer1Size; c++)
                                    neu1[c] += _syn0[c + lastWord * _layer1Size];
                                cw++;
                            }

                        if (cw > 0)
                        {
                            for (c = 0; c < _layer1Size; c++)
                                neu1[c] /= cw;

                            if (_hs > 0)
                                for (d = 0; d < _vocab[word].CodeLen; d++)
                                {
                                    f = 0;

                                    l2 = _vocab[word].Point[d] * _layer1Size;

                                    // Propagate hidden -> output
                                    for (c = 0; c < _layer1Size; c++)
                                        f += neu1[c] * _syn1[c + l2];

                                    if (f <= MaxExp * -1)
                                        continue;

                                    if (f >= MaxExp)
                                        continue;

                                    f = _expTable[(int)((f + MaxExp) * (ExpTableSize / MaxExp / 2))];

                                    // 'g' is the gradient multiplied by the learning rate
                                    g = (1 - _vocab[word].Code[d] - f) * _alpha;

                                    // Propagate errors output -> hidden
                                    for (c = 0; c < _layer1Size; c++)
                                        neu1e[c] += g * _syn1[c + l2];

                                    // Learn weights hidden -> output
                                    for (c = 0; c < _layer1Size; c++)
                                        _syn1[c + l2] += g * neu1[c];

                                }

                            // NEGATIVE SAMPLING
                            if (_negative > 0)
                                for (d = 0; d < _negative + 1; d++)
                                {
                                    if (d == 0)
                                    {
                                        target = word;
                                        label = 1;
                                    }
                                    else
                                    {
                                        nextRandom = nextRandom * 25214903917 + 11;

                                        target = _table[(nextRandom >> 16) % TableSize];

                                        if (target == 0)
                                            target = (long)(nextRandom % (ulong)(_vocabSize - 1) + 1);

                                        if (target == word)
                                            continue;

                                        label = 0;
                                    }

                                    l2 = target * _layer1Size;

                                    f = 0;

                                    for (c = 0; c < _layer1Size; c++)
                                        f += neu1[c] * _syn1Neg[c + l2];

                                    if (f > MaxExp)
                                        g = (label - 1) * _alpha;
                                    else if (f < MaxExp * -1)
                                        g = (label - 0) * _alpha;
                                    else
                                        g = (label - _expTable[(int)((f + MaxExp) * (ExpTableSize / MaxExp / 2))]) * _alpha;

                                    for (c = 0; c < _layer1Size; c++)
                                        neu1e[c] += g * _syn1Neg[c + l2];

                                    for (c = 0; c < _layer1Size; c++)
                                        _syn1Neg[c + l2] += g * neu1[c];

                                }

                            // hidden -> in
                            for (var a = b; a < _window * 2 + 1 - b; a++)
                                if (a != _window)
                                {
                                    c = sentencePosition - _window + a;

                                    if (c < 0)
                                        continue;

                                    if (c >= sentenceLength)
                                        continue;

                                    lastWord = sen[c];

                                    if (lastWord == -1)
                                        continue;

                                    for (c = 0; c < _layer1Size; c++)
                                        _syn0[c + lastWord * _layer1Size] += neu1e[c];

                                }

                        }

                    }
                    else
                    {
                        //train skip-gram
                        for (var a = b; a < _window * 2 + 1 - b; a++)
                            if (a != _window)
                            {
                                c = sentencePosition - _window + a;

                                if (c < 0)
                                    continue;
                                if (c >= sentenceLength)
                                    continue;

                                lastWord = sen[c];

                                if (lastWord == -1)
                                    continue;

                                var l1 = lastWord * _layer1Size;

                                for (c = 0; c < _layer1Size; c++)
                                    neu1e[c] = 0;

                                // HIERARCHICAL SOFTMAX
                                if (_hs != 0)
                                    for (d = 0; d < _vocab[word].CodeLen; d++)
                                    {
                                        f = 0;

                                        l2 = _vocab[word].Point[d] * _layer1Size;

                                        // Propagate hidden -> output
                                        for (c = 0; c < _layer1Size; c++)
                                            f += _syn0[c + l1] * _syn1[c + l2];

                                        if (f <= MaxExp * -1)
                                            continue;

                                        if (f >= MaxExp)
                                            continue;

                                        f = _expTable[(int)((f + MaxExp) * (ExpTableSize / MaxExp / 2))];

                                        // 'g' is the gradient multiplied by the learning rate
                                        g = (1 - _vocab[word].Code[d] - f) * _alpha;

                                        // Propagate errors output -> hidden
                                        for (c = 0; c < _layer1Size; c++)
                                            neu1e[c] += g * _syn1[c + l2];

                                        // Learn weights hidden -> output
                                        for (c = 0; c < _layer1Size; c++)
                                            _syn1[c + l2] += g * _syn0[c + l1];

                                    }
                                // NEGATIVE SAMPLING
                                if (_negative > 0)
                                    for (d = 0; d < _negative + 1; d++)
                                    {
                                        if (d == 0)
                                        {
                                            target = word;
                                            label = 1;
                                        }
                                        else
                                        {
                                            nextRandom = nextRandom * 25214903917 + 11;

                                            target = _table[(nextRandom >> 16) % TableSize];

                                            if (target == 0)
                                                target = (long)(nextRandom % (ulong)(_vocabSize - 1) + 1);

                                            if (target == word)
                                                continue;

                                            label = 0;
                                        }

                                        l2 = target * _layer1Size;

                                        f = 0;

                                        for (c = 0; c < _layer1Size; c++)
                                            f += _syn0[c + l1] * _syn1Neg[c + l2];

                                        if (f > MaxExp)
                                            g = (label - 1) * _alpha;
                                        else if (f < MaxExp * -1)
                                            g = (label - 0) * _alpha;
                                        else
                                            g = (label - _expTable[(int)((f + MaxExp) * (ExpTableSize / MaxExp / 2))]) * _alpha;

                                        for (c = 0; c < _layer1Size; c++)
                                            neu1e[c] += g * _syn1Neg[c + l2];

                                        for (c = 0; c < _layer1Size; c++)
                                            _syn1Neg[c + l2] += g * _syn0[c + l1];

                                    }

                                // Learn weights input -> hidden
                                for (c = 0; c < _layer1Size; c++)
                                    _syn0[c + l1] += neu1e[c];

                            }

                    }

                    sentencePosition++;

                    if (sentencePosition >= sentenceLength)
                    {
                        sentenceLength = 0;
                    }

                }

            }

            neu1 = null;

            neu1e = null;

            GC.Collect();

        }



        /// <summary>
        /// Train a Model with the previously specified paramteters on the specified Corpus. 
        /// The Model is used by the other classes to search an compare the Word Vectors.
        /// </summary>
        public void TrainModel()
        {

            long d;

            Console.WriteLine("Starting training using file {0}\n", _trainFile);

            _startingAlpha = _alpha;

            if (!string.IsNullOrEmpty(_readVocabFile))
                ReadVocab();
            else
                LearnVocabFromTrainFile();

            if (!string.IsNullOrEmpty(_saveVocabFile))
                SaveVocab();

            if (string.IsNullOrEmpty(_outputFile))
                return;

            InitNet();

            if (_negative > 0)
                InitUnigramTable();

            _stopwatch = new Stopwatch();
            _stopwatch.Start();


            ParallelOptions parallelOptions = new ParallelOptions()
            {
                MaxDegreeOfParallelism = _numThreads
            };

            var result = Parallel.For(0, _numThreads, parallelOptions, TrainModelThread);

            if (!result.IsCompleted)
            {
                throw new InvalidOperationException();
            }

            //TrainModelThreadStart(1);
            using (var stream = new FileStream(_outputFile, FileMode.Create, FileAccess.Write))
            {

                long b;

                if (_classes == 0)
                {

                    // Save the word vectors
                    var bytes = string.Format("{0} {1}\n", _vocabSize, _layer1Size).GetBytes();

                    stream.Write(bytes, 0, bytes.Length);

                    for (var a = 0; a < _vocabSize; a++)
                    {

                        bytes = string.Concat(_vocab[a].Word, ' ').GetBytes();

                        stream.Write(bytes, 0, bytes.Length);

                        if (_binary > 0)
                        {

                            for (b = 0; b < _layer1Size; b++)
                            {

                                bytes = BitConverter.GetBytes(_syn0[a * _layer1Size + b]);

                                stream.Write(bytes, 0, bytes.Length);
                            }
                        }

                        else
                        {

                            for (b = 0; b < _layer1Size; b++)
                            {

                                bytes = string.Concat(_syn0[a * _layer1Size + b], " ").GetBytes();

                                stream.Write(bytes, 0, bytes.Length);
                            }
                        }

                        bytes = "\n".GetBytes();

                        stream.Write(bytes, 0, bytes.Length);
                    }
                }
                else
                {
                    // Run K-means on the word vectors
                    int clcn = (int)_classes, iter = 10;

                    var centcn = new int[_classes];

                    var cl = new int[_vocabSize];

                    var cent = new float[_classes * _layer1Size];

                    for (var a = 0; a < _vocabSize; a++) cl[a] = a % clcn;

                    for (var a = 0; a < iter; a++)
                    {
                        for (b = 0; b < clcn * _layer1Size; b++)
                            cent[b] = 0;

                        for (b = 0; b < clcn; b++)
                            centcn[b] = 1;

                        long c;

                        for (c = 0; c < _vocabSize; c++)
                        {

                            for (d = 0; d < _layer1Size; d++)
                                cent[_layer1Size * cl[c] + d] += _syn0[c * _layer1Size + d];

                            centcn[cl[c]]++;
                        }

                        float closev;

                        for (b = 0; b < clcn; b++)
                        {

                            closev = 0;

                            for (c = 0; c < _layer1Size; c++)
                            {

                                cent[_layer1Size * b + c] /= centcn[b];

                                closev += cent[_layer1Size * b + c] * cent[_layer1Size * b + c];
                            }

                            closev = (float)Math.Sqrt(closev);

                            for (c = 0; c < _layer1Size; c++)
                                cent[_layer1Size * b + c] /= closev;

                        }

                        for (c = 0; c < _vocabSize; c++)
                        {
                            closev = -10;

                            var closeid = 0;

                            for (d = 0; d < clcn; d++)
                            {
                                float x = 0;

                                for (b = 0; b < _layer1Size; b++)
                                    x += cent[_layer1Size * d + b] * _syn0[c * _layer1Size + b];

                                if (x > closev)
                                {

                                    closev = x;

                                    closeid = (int)d;
                                }

                            }

                            cl[c] = closeid;
                        }

                    }

                    // Save the K-means classes
                    for (var a = 0; a < _vocabSize; a++)
                    {
                        var bytes = string.Format("{0} {1}\n", _vocab[a].Word, cl[a]).GetBytes();

                        stream.Write(bytes, 0, bytes.Length);
                    }

                    centcn = null;
                    cent = null;
                    cl = null;

                }

            }

            GC.Collect();

        }

    }
}
