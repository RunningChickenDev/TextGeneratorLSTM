import json

class MetaData:
	def __init__(self):
		self.source = ""
		self.data = ""
		self.n_chars = 0
		self.n_vocab = 0
		self.char_to_int = {}
		self.int_to_char = {}
		self.sequenceLength = 100
		self.modelType = "double"
		self.name = ""
		self.authors = []

def readData(filename):
	m = MetaData()
	obj = {}
	with open(filename, 'r') as f:
		obj = json.load(f)

	try:
		m.source = obj['source']
		m.data = obj['data']
		m.n_chars = obj['n_chars']
		m.n_vocab = obj['n_vocab']
		m.char_to_int = obj['char_to_int']
		m.int_to_char = obj['int_to_char']
		m.sequenceLength = obj['sequenceLength']
		m.modelType = obj['modelType']
	except KeyError:
		print("Could not read JSON data load! Returning None")
		return None

	try:
		m.name = obj['name']
		m.authors = obj['authors']
	except:
		pass

	return m

def prepareData(inputtext, outputdata, outputmetadata, cast=None):
	print("Preparing data from {}, to {} and {} with cast {}"
			.format(inputtext, outputdata, outputmetadata, cast))
	extractData(inputtext, outputdata, cast=cast)
	extractMetaData(inputtext, outputdata, outputmetadata, cast=cast)

def extractData(inputtext, outputdata, cast=None):
	print("Extracting data from {} to {}".format(inputtext, outputdata))

	# Get cast
	m = None
	if cast != None:
		print("\tReading cast")
		m = readData(cast)

	# Extract data
	f_in = open(inputtext, "r")
	f_out = open(outputdata, "w")
	print("\tOpened files")
	for line in f_in.readlines():
		for c in line:
			# Is there previous metadata?
			if m != None:
				if c.lower() in m.char_to_int.keys():
					f_out.write(c.lower())
			# Is it ASCII?
			else:
				if ord(c) < 128:
					f_out.write(c.lower())
	f_in.close()
	f_out.close()
	print("\tClosed files")

def extractMetaData(inputtext, inputdata, outputmetadata, name=None, authors=None, cast=None):
	print("Extracting metadata from {} and {} to {}".format(inputtext, inputdata, outputmetadata))

	# Create MetaData
	m = None
	if cast != None:
		m = readData(cast)
	else:
		m = MetaData()

	# Set predetermined variables
	m.source = inputtext
	m.data = inputdata

	# Set optional parameters
	if name != None:
		m.name = name
	if authors != None:
		m.authors = authors

	# Actual extraction
	print("\tActual extraction")
	raw_text = open(inputdata).read()
	if cast == None:
		chars = sorted(list(set(raw_text)))
		m.char_to_int = dict((c, i) for i, c in enumerate(chars))
		m.int_to_char = dict((i, c) for i, c in enumerate(chars))
		m.n_chars = len(raw_text)
		m.n_vocab = len(chars)
		# # default value
		# m.sequenceLength = 100
		# m.modelType = "double"
	else:
		m.n_chars = len(raw_text)
		m.n_vocab = len(m.int_to_char.keys())

	with open(outputmetadata, 'w') as f:
		print("\tWriting file")
		json.dump(vars(m), f, sort_keys=True, indent=4)

def generateGenericMetaData(outputmetadata):
	obj = MetaData()
	obj.name = "DataSet"
	obj.authors = ["Generic User"]
	obj.source = "dataset.txt"
	obj.data = "dataset.data.txt"

	obj.char_to_int = {}
	obj.int_to_char = {}
	obj.n_chars = -1

	for i in range(0, 26):
		obj.int_to_char[i] = chr(97+i)	# lower case a-z

	for i in range(26, 36):
		obj.int_to_char[i] = "" + str(i-26)	# lower case a-z

	# reading characters
	obj.int_to_char[36] = " "
	obj.int_to_char[37] = ","
	obj.int_to_char[38] = "."
	obj.int_to_char[39] = ";"
	obj.int_to_char[40] = ":"
	obj.int_to_char[41] = "("
	obj.int_to_char[42] = ")"
	obj.int_to_char[43] = "-"
	obj.int_to_char[44] = "'"
	obj.int_to_char[45] = "\""
	obj.int_to_char[46] = "\n"
	obj.int_to_char[47] = "\t"
	obj.int_to_char[48] = "!"
	obj.int_to_char[49] = "?"

	obj.n_vocab = len(obj.int_to_char.keys())

	for i in range(0, obj.n_vocab):
		obj.char_to_int[obj.int_to_char[i]] = i

	with open(outputmetadata, 'w') as f:
		json.dump(vars(obj), f, sort_keys=True, indent=4)

if __name__ == '__main__':
	# generateGenericMetaData("generic.data.json")
	prepareData("dataset_bnw_en.data.txt", "bnw/bnw.data.txt", "bnw/bnw.data.json", cast="generic.data.json")
	# extractMetaData("BNW chp. 1-6", "bnw/bnw.data.json", "../dataset_bnw_en.data.txt")
