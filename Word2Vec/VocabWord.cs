namespace Word2Vec
{



    #region Using Statements:



    using System;
    using System.Collections.Generic;



    #endregion



    /// <summary>
    /// Declared as a struct in the orriginal C Code as a store for the Word.
    /// </summary>
    internal struct VocabWord : IComparable<VocabWord>
    {

        public long Cn { get; set; }


        public string Word { get; set; }


        public char[] Code { get; set; }


        public int CodeLen { get; set; }


        public int[] Point { get; set; }


        public int CompareTo(VocabWord other)
        {
            return (int)(this.Cn - other.Cn);
        }

    }



    /// <summary>
    /// Used in: SortVocab()
    /// Use for Sorting by Word Count.
    /// </summary>
    internal class VocubComparer : IComparer<VocabWord>
    {

        public int Compare(VocabWord x, VocabWord y)
        {
            return (int)(y.Cn - x.Cn);
        }

    }

}