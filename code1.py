import argparse

if __name__ == "__main__":  # if this script was run through "python code1.py"
    parser = argparse.ArgumentParser(description = 'PySpark Dictionary Learning',
        add_help = 'How to use', prog = 'python code1.py <args>')

    # Required parameters.
    parser.add_argument("-i", "--input", required = True,
        help = "File name.")
    parser.add_argument("-l", "--length", type = int, required = True,
        help = "Length of the samples.")

    # Optional parameters.
    parser.add_argument("-nz", "--nonzero", type = float, default = 0.7,
        help = "Percentage of non-zero elements. [DEFAULT: 0.7]")

    # Parses the arguments.
    args = vars(parser.parse_args())

    # All command line arguments are now available through args dictionary,
    # accessed in this way:
    #    args['input'] # the string input path
    #    args['nonzero'] # the floating-point nonzero elements

flag = True
while flag: 
	file_S = input('1. Please Enter the File name:')
	t = input('2. Please Enter the length of samples:')
	p = input ('3. Please enter the number of samples:')
	file_D = input('4. Please Enter the dictioary file name:')
	file_Z = input('5. Please Enter the output loading file name:')
	file_summary = input('6. Please Enter the summary File name:')
	m = input('7. Please Enter the the number of dictionaty atoms')
	r = ('8. Please Enter the percentage of non-zero elements')
	epsilon = input('9. Please Enter a value for epsilon')
	print('All needed information are imported')
	answ = input ('Are the above information completely correct? (y/n)')
	if answ == 'n':
		n = int(input('which information is incorrect? ( Enter a number between 1 to 9 ):'))
		if n == 1:
		 	file_S = input('1. Please Enter the File name:')
		 	flag = False
		elif n == 2:
   			t = input(' 2. Please Enter the length of samples:')
   			flag = False
   		elif n == 3:
   			p = input(' 3. Please Enter the number of samples:')
   			flag = False
   		elif n == 4:
   			file_D = input('4. Please Enter the dictioary file name:')
   			flag = False
   		elif n == 5:
   			file_Z = input('5. Please Enter the output loading file name:')
   			flag = False
   		elif n == 6:
   			file_summary = input('6. Please Enter the summary File name:')
   			flag = False
   		elif n == 7:
   			m = input('7. Please Enter the the number of dictionaty atoms')
   			flag = False
   		elif n == 8:
   			r = ('8. Please Enter the percentage of non-zero elements')
   			flag = False
   		elif n == 9:
   			epsilon = input('9. Please Enter a value for epsilon')
   			flag = False

   	else:
   		flag = False
   	 	