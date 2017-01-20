namespace WordAnalogy
{



    #region Using Statements:



    using System;
    using System.IO;

    using Utils;



    #endregion



    public class WordAnalogy
    {



        #region Fields:



        // Max Length of Strings:
        public const long max_size = 2000;

        // The Number of closest Words that will be shown:
        public long N = 40;

        // The Max length of vocabulary entries:
        public const long max_w = 50;

        private long size;

        // File Name and Path:
        private string file_name;

        private long words;

        private char[] vocab;


        /// <summary>
        /// // M a complete Float Array of all Vectors, defined by the Words(234868) Multiplied by the Size(200)
        /// </summary>
        private float[] m;



        #endregion



        #region Properties:



        /// <summary>
        /// The Vocabulary in the form of a Char Array.
        /// Note: variable 'max_w' defines each word and if the word is 3 chars long, then the remaining chars are filled with '\0'
        /// </summary>
        protected char[] Vocab
        {
            get
            {
                return vocab;
            }

            set
            {
                vocab = value;
            }
        }



        /// <summary>
        /// the total number of the Words.
        /// </summary>
        protected long Words
        {
            get
            {
                return words;
            }

            set
            {
                words = value;
            }
        }



        /// <summary>
        /// The Size of the Vectors for each word.
        /// </summary>
        protected long Size
        {
            get
            {
                return size;
            }

            set
            {
                size = value;
            }
        }



        /// <summary>
        /// // M a complete Float Array of all Vectors, defined by the Words(234868) Multiplied by the Size(200)
        /// </summary>
        protected float[] M
        {
            get
            {
                return m;
            }

            set
            {
                m = value;
            }
        }




        #endregion



        /// <summary>
        /// Word Analogy class.
        /// Tomas Mikolov - Note that for the word analogy to perform well, the model should be trained on much larger data set Example input: "paris france berlin"
        /// Usage in Demo: 1: Train into 'vectors.bin' 2: Load 'vectors.bin' into the word_analogy Class. 
        /// 1: './word2vec -train text8 -output vectors.bin -cbow 1 -size 200 -window 8 -negative 25 -hs 0 -sample 1e-4 -threads 20 -binary 1 -iter 15' 
        /// 2: './word-analogy vectors.bin'
        /// Where: wget http://mattmahoney.net/dc/text8.zip
        /// </summary>
        /// <param name="fileName">File must be a correctly formatted bin file. This option is not on by default.</param>
        public WordAnalogy(string fileName)
        {

            // Initialise the File Name and Path:
            file_name = fileName;

            // Load Vocabulary:
            InitVocub();

        }



        /// <summary>
        /// Initialise and Load the Vocabulary.
        /// File must be a correctly formatted bin file. This option is not on by default.
        /// </summary>
        private void InitVocub()
        {

            using (FileStream f = File.Open(file_name, FileMode.Open, FileAccess.Read))
            {
                // Read the Words Size, normally a lrge number: 234868?
                Words = f.ReadInt32();

                // Read the Size, this is normally a smaller number: 200?
                Size = f.ReadInt32();

                // M a complete Float Array of all Vectors, defined by the Words(234868) Multiplied by the Size(200)
                M = new float[Words * Size];

                // The Vocab, this is a Char Array of the chars of each word, separated by the distance of 50 Chars at a time. This is max_w = 50  max length of vocabulary entries
                Vocab = new char[Words * max_w];

                for (int b = 0; b < Words; b++)
                {

                    int a = 0;

                    string word = f.ReadWord();

                    foreach (char ch in word)
                    {
                        Vocab[b * max_w + a] = ch;

                        if ((a < max_w) && (vocab[b * max_w + a] != '\n'))
                            a++;
                    }

                    Vocab[b * max_w + a] = '\0';

                    for (a = 0; a < Size; a++)
                    {

                        byte[] bytes = new byte[4];

                        f.Read(bytes, 0, 4);

                        M[a + b * Size] = BitConverter.ToSingle(bytes, 0);
                    }

                    float len = 0;

                    for (a = 0; a < Size; a++)
                        len += M[a + b * Size] * M[a + b * Size];

                    len = (float)Math.Sqrt(len);

                    for (a = 0; a < Size; a++)
                        M[a + b * Size] = M[a + b * Size] / len;

                }

            }

        }



        /// <summary>
        /// Search nearest words Analogy to the searchtext parameter.
        /// Example: "paris france berlin" 
        /// Answer should be: "Germany"
        /// </summary>
        /// <param name="searchtext">The Input Text to Search.</param>
        /// <param name="results">The Number of Results to be returned.</param>
        public void Search(string searchtext, long results)
        {

            // Initialise a new BestWords Array:
            BestWord[] bestWords = new BestWord[N];

            // An Array of Longs, the position in the Vocabulary:
            long[] bi = new long[100];

            // Vector result of Vector Math:
            float[] vec = new float[max_size];

            // The searchtext Split into a String Array:
            string[] st = searchtext.Split(' ');

            // cn is the searchtext String.Split Length:
            long cn = st.Length;

            // The number of results to return:
            N = results;

            // Check words were entered:
            if (cn < 3)
            {
                Console.WriteLine("Only " + cn + " words were entered.. three words are needed at the input to perform the calculation");

                return;
            }

            // The Inswx od the each Word in the for loop.
            long b = -1;

            // Loop through the Strings in the searchtext Split (st):
            for (long a = 0; a < cn; a++)
            {

                // Find the Word in the Vocabulary:
                for (b = 0; b < Words; b++)
                {
                    // Look Up the Word:
                    string word = new string(Vocab, (int)(b * max_w), (int)max_w).Replace("\0", string.Empty);

                    // Found, break and use b as the Index:
                    if (word.Equals(st[a]))
                        break;
                }

                // End of Words (Count):
                if (b == Words) b = -1;

                // Add the Position to bi[a] this is an Index:
                bi[a] = b;

                // Display Word Location in the Vocabulary:
                Console.WriteLine("Word: {0}  Position in vocabulary: {1}", st[a], bi[a]);

                // Word not found:
                if (b == -1)
                {
                    Console.WriteLine("Out of dictionary word!");
                    break;
                }

            }

            // We want b to not be -1:
            if (b == -1)
                return;



            // Write the Header to the Console:
            Console.WriteLine("{0,50}{1,22}\n{2}", "Word", "Cosine distance", new string('-', 72));



            // This is the Magic:
            // M is the Vectors, and we are doing Math on the Vectors for Each Word:
            for (long a = 0; a < Size; a++)
                vec[a] = M[a + bi[1] * size] - M[a + bi[0] * size] + M[a + bi[2] * size];


            // Vector Magnitude Normalisation: Unit Vector
            float len = 0;
            for (long a = 0; a < Size; a++)
                len += vec[a] * vec[a];
            len = (float)Math.Sqrt(len);
            for (long a = 0; a < Size; a++)
                vec[a] /= len;


            // Process the Vectors:
            for (long c = 0; c < Words; c++)
            {
                long a = 0;

                for (b = 0; b < cn; b++)
                    if (bi[b] == c)
                        a = 1;

                if (a == 1)
                    continue;

                float dist = 0;

                for (a = 0; a < Size; a++)
                    dist += vec[a] * M[a + c * Size];

                for (a = 0; a < N; a++)
                {
                    if (dist > bestWords[a].Distance)
                    {
                        for (long d = N - 1; d > a; d--)
                        {
                            bestWords[d] = bestWords[d - 1];
                        }

                        bestWords[a].Distance = dist;

                        bestWords[a].Word = new string(Vocab, (int)(max_w * c), (int)max_w).Replace("\0", String.Empty).Trim();

                        break;
                    }

                }

            }

            // Print the Result:
            for (int a = 0; a < N; a++)
                Console.WriteLine("{0,50}{1,20}", bestWords[a].Word, bestWords[a].Distance);

        }

    }

}
