


# ============= YOUR CODE HERE ==============
#Runs with python3 on coderunner-SEPT-2017
#code creates a file called gender_degree_data.tsv that contains all data about the gender breakdowns of various degree majors in the US.
#===========================================
import requests
from bs4 import BeautifulSoup

with open('gender_degree_data.tsv', 'w') as out_file:
	out_file.write('\t'.join(['Year', 'Degree_Major','Total_Bachelors','Percent_change_Bachelors','Male_Bachelors', 'Female_Bachelors', 'Female_percent_Bachelors','Total_Masters', 'Male_Masters', 'Female_Masters','Total_Doctorates','Male_Doctorates', 'Female_Doctorates']) + '\n')

	r = requests.get('http://nces.ed.gov/programs/digest/current_tables.asp')
	#print(r.url)
	soup = BeautifulSoup(r.text,"html.parser")
	#print(soup.prettify())
	for link in soup.find_all('a', href=True):
		if 'dt16_325' in link.get("href"):
			#print(link.get("href"))
			url = 'http://nces.ed.gov/programs/digest/{}'.format(link['href'])
			url_response = requests.get(url)
			url_response = BeautifulSoup(url_response.text, "html.parser")
			degree_major = url_response.find('title').text.split('Degrees in')[1].split('conferred')[0].strip()
			#print(degree_major)
			all_trs = url_response.find_all('tr')
			for tr in all_trs:
								# We only want to parse entries that correspond to a certain year
				year_header = tr.find('th')
				if year_header is None:
					continue

								# Stop parsing after all of the years are listed
				if 'Percent change' in year_header.text:
					break

								# Years always have a dash (-) in them
				if '-' not in year_header.text:
					continue

				year = str(int(year_header.text.split('-')[0]) + 1)
				year_vals = [x.text.replace(',', '').replace('†', '0').replace('#', '0') for x in tr.find_all('td')]

				out_text = '\t'.join([year, degree_major] + year_vals) + '\n'
			out_file.write(out_text)




			



