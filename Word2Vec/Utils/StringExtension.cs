namespace Utils
{




    #region Using Statetents:



    using System.Text;



    #endregion




    /// <summary>
    /// String Extension class.
    /// </summary>
    public static class StringExtension
    {



        /// <summary>
        /// Converts a String to a Byte[].
        /// </summary>
        /// <param name="input">The String input.</param>
        /// <returns>Byte[] that represents the String.</returns>
        public static byte[] GetBytes(this string input)
        {
            return Encoding.UTF8.GetBytes(input);
            //byte[] bytes = new byte[str.Length * sizeof(char)];
            //System.Buffer.BlockCopy(str.ToCharArray(), 0, bytes, 0, bytes.Length);
            //return bytes;
        }



        /// <summary>
        /// Converts a Byte[] to a String.
        /// </summary>
        /// <param name="bytes">The String input.</param>
        /// <returns>String that represents the Byte[].</returns>
        public static string GetString(this byte[] bytes)
        {
            char[] chars = new char[bytes.Length / sizeof(char)];
            System.Buffer.BlockCopy(bytes, 0, chars, 0, bytes.Length);
            return new string(chars);
        }



        /// <summary>
        /// Cleans a String that has unwanted Chars.
        /// </summary>
        /// <param name="word">The String input.</param>
        /// <returns>String that represents the String.</returns>
        public static string Clean(this string word)
        {
            return word.Replace(".", "").Replace(",", "").Replace("!", "").Replace("?", "").Replace("\"", "").ToLower().Trim();
        }



    }

}
