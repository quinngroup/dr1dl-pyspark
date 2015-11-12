
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
   	 	