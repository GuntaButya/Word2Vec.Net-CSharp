namespace GUI
{




    #region Using Statements:



    using System;

    using Distance;

    using WordAnalogy;



    #endregion



    class Program
    {



        #region Fields:



        static string path = @"Word2VectorOutputFile.bin";



        static Distance distance = null;



        static WordAnalogy wordAnalogy = null;



        #endregion



        #region Properties:




        #endregion



        static void Main(string[] args)
        {


            // -train <file> Use text data from <file> to train the model
            string train = "Corpus.txt";

            // -output <file> Use <file> to save the resulting word vectors / word clusters
            string output = "Vectors.bin";

            // -save-vocab <file> The vocabulary will be saved to <file>
            string savevocab = "";

            // -read-vocab <file> The vocabulary will be read from <file>, not constructed from the training data
            string readvocab = "";

            // -size <int> Set size of word vectors; default is 100
            int size = 100;

            // -debug <int> Set the debug mode (default = 2 = more info during training)
            int debug = 2;

            // -binary <int> Save the resulting vectors in binary moded; default is 0 (off)
            int binary = 1;

            // -cbow <int> Use the continuous bag of words model; default is 1 (use 0 for skip-gram model)
            int cbow = 1;

            // -alpha <float> Set the starting learning rate; default is 0.025 for skip-gram and 0.05 for CBOW
            float alpha = 0.05f;

            // -sample <float> Set threshold for occurrence of words. Those that appear with higher frequency in the training data
            float sample = 1e-4f;

            // -hs <int> Use Hierarchical Softmax; default is 0 (not used)
            int hs = 0;

            // -negative <int> Number of negative examples; default is 5, common values are 3 - 10 (0 = not used)
            int negative = 25;

            // -threads <int> Use <int> threads (default 12)
            int threads = 12;

            // -iter <int> Run more training iterations (default 5)
            long iter = 15;

            // -min-count <int> This will discard words that appear less than <int> times; default is 5
            int mincount = 5;

            // -classes <int> Output word classes rather than word vectors; default number of classes is 0 (vectors are written)
            long classes = 0;

            // -window <int> Set max skip length between words; default is 5
            int window = 12;

            Word2Vec.Word2Vec word2Vec = new Word2Vec.Word2Vec(train, output, savevocab, readvocab, size, debug, binary, cbow, alpha, sample, hs, negative, threads, iter, mincount, classes, window);


            Console.WriteLine("Train Model: Y/N");
            string usr = Console.ReadLine().ToLower();
            if(usr =="y")
                word2Vec.TrainModel();


            path = @"Vectors.bin";
            distance = new Distance(path);
            wordAnalogy = new WordAnalogy(path);
            

            while (true)
            {

                Console.Clear();

                Console.WriteLine("{0}\nPlease enter Word: ", new string('-', 50));

                string input = Console.ReadLine();

                Console.Clear();

                distance.Search(input, 10);

                wordAnalogy.Search(input, 10);

                Console.ReadLine();

            }

        }

    }

}
