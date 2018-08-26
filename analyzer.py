import csv
import io
import re
import sys


def analyze_freq_char(original="", output="", force_original_chars=True):
	freq = {}
	freq['original'] = {'corpus length':0}
	with open(original, "r") as f:
		for line in f:
			for c in line:
				freq['original']['corpus length'] += 1
				c = c.lower()
				try:
					freq['original'][c] += 1
				except:
					freq['original'][c] = 1

	freq['output'] = {'corpus length':0}
	if force_original_chars:
		for key in freq['original'].keys():
			freq['output'][key] = 0
	with open(output, "r") as f:
		for line in f:
			for c in line:
				freq['output']['corpus length'] += 1
				c = c.lower()
				try:
					freq['output'][c] += 1
				except:
					if not force_original_chars:
						freq['output'][c] = 1
	return freq

def analyze_freq_syll(original="", output="", sylls=[]):
	if not sylls:
		return {}
	freq = {}
	dor = io.open(original).read()
	dou = io.open(output).read()
	freq['original'] = {'corpus length': len(dor)}
	freq['output'] = {'corpus length': len(dou)}
	for syll in sylls:
		freq['original'][syll] = len(re.findall(syll, dor))
		freq['output'][syll] = len(re.findall(syll, dou))

	return freq

def analyze_freq_word(original="", output=""):
	words = []
	freq = {'original': {}, 'output': {}}
	tor = io.open(original, encoding='ascii').read().lower()
	wor = re.split(r'([^\w]+)', tor)
	print("{} words in original".format(len(wor)))
	for word in wor:
		if word not in words:
			sys.stdout.write('\r{}'.format(len(words)))
			sys.stdout.flush()
			words += [word]
		try:
			freq['original'][word] += 1
		except:
			freq['original'][word] = 1
	print()
	print("{} different words registered".format(len(words)))
	tou = io.open(output, encoding='ascii').read().lower()
	wou = re.split(r'([^\w]+)', tou)
	print("{} words in output".format(len(wou)))
	for word in wou:
		if word in words:
			try:
				freq['output'][word] += 1
			except:
				freq['output'][word] = 1
	freq['words'] = words
	freq['original']['word count'] = len(words)
	freq['output']['word count'] = len(words)

	return freq

if __name__ == '__main__':
	def_sylls = [
		"aa",
		"ee",
		"uu",
		"ui",
		"ij",
		"sch",
		"wh",
		"ai",
		"oi",
		"ck",
		"xc",
		"tch",
		"ei",
		"oo",
		"ou",
		"ue"
	]

	print (" GIVE DATA PLS ")
	data = analyze_freq_word(original = "obabo/in.txt", output = "obabo/written2.mind.txt")
	print (" O K DATA ")
	with open("obabo/data/word_oor.csv", "w+") as f:
		csvw = csv.writer(f)
		for word in data['original'].keys():
			csvw.writerow([word, data['original'][word]])
	with open("obabo/data/word_oou.csv", "w+") as f:
		csvw = csv.writer(f)
		for word in data['output'].keys():
			csvw.writerow([word, data['output'][word]])

	# print("** Full analysis **")
	# print("Original:")
	# original = input("> ")
	# print("Output:")
	# output = input("> ")
	# print("* Character Frequency *")
	# data = analyze_freq_char(original=original, output=output)
	# print("Data file for original:")
	# out_original = input("> ")
	# with open(out_original, "w") as f:
	# 	csvw = csv.writer(f)
	# 	for char in data['original'].keys():
	# 		csvw.writerow([char, data['original'][char]])
	# print("Data file for output:")
	# out_output = input("> ")
	# with open(out_output, "w") as f:
	# 	csvw = csv.writer(f)
	# 	for char in data['output'].keys():
	# 		csvw.writerow([char, data['output'][char]])
	# print("* Syllable Frequency *")
	# print("Syllables (empty for default):")
	# raw_sylls = input("> ")
	# sylls = None
	# if not raw_sylls:
	# 	sylls = list(def_sylls)
	# else:
	# 	sylls = re.split(r'\W+', raw_sylls)
	# data = analyze_freq_syll(original, output, sylls)
	# print("Data file for original:")
	# out_original = input("> ")
	# with open(out_original, "w") as f:
	# 	csvw = csv.writer(f)
	# 	for syll in data['original'].keys():
	# 		csvw.writerow([syll, data['original'][syll]])
	# print("Data file for output:")
	# out_output = input("> ")
	# with open(out_output, "w") as f:
	# 	csvw = csv.writer(f)
	# 	for syll in data['output'].keys():
	# 		csvw.writerow([syll, data['output'][syll]])
