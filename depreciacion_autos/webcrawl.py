from html.parser import HTMLParser
from urllib.error import HTTPError
import urllib.request
from urllib.request import urlopen
from urllib import parse
import sys
import pandas as pd


url = 'https://www.yapo.cl/chile/autos?ca=5_s&l=0&st=s&br=68&mo=36'

url_2 = 'https://www.chileautos.cl/autos/busqueda?s={:d}&q=(C.Marca.Peugeot._.Modelo.2008.)&l={:d}'


class LinkParser(HTMLParser):

    def __init__(self):
        HTMLParser.__init__(self)
        self.in_tbody = False
        self.keep_going = True
        self.interesting_data = []
    # This is a function that HTMLParser normally has
    # but we are adding some functionality to it
    def handle_starttag(self, tag, attrs):
        # We are looking for the begining of a link. Links normally look
        # like <a href="www.someurl.com"></a>
        if tag =='table':
            self.in_tbody = True

        if tag == 'a':
            for (key, value) in attrs:
                if key == 'href':
                    # We are grabbing the new URL. We are also adding the
                    # base URL to it. For example:
                    # www.netinstructions.com is the base and
                    # somepage.html is the new URL (a relative URL)
                    #
                    # We combine a relative URL with the base URL to create
                    # an absolute URL like:
                    # www.netinstructions.com/somepage.html
                    if self.search in value:
                        newUrl = parse.urljoin(self.baseUrl, value)
                        # And add it to our colection of links:
                        if newUrl not in self.links:
                            self.links = self.links + [newUrl]

    def handle_endtag(self, tag):
        if tag == 'table':
            self.in_tbody = False
            self.keep_going = False

    def handle_data(self, data):
        if self.in_tbody and self.keep_going:
            if '  ' not in data and ' Cargos' not in data:
                self.interesting_data.append(data)
            elif ' Cargos' in data:
                print('###########')
                print(data)
                print('###########')
                self.reset()

            # filtered_data = data.split('Cargos')
            # print(filtered_data[0])


    # This is a new function that we are creating to get links
    # that our spider() function will call
    def getLinks(self, url, search, base_url):
        self.links = []
        # Remember the base URL which will be important when creating
        # absolute URLs
        self.baseUrl = base_url
        # Use the urlopen function from the standard Python 3 library
        try:
            req = urllib.request.Request(url,
                                         data=None,
                                         headers={
                                             'User-Agent': 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36'})
            response = urlopen(req)
            # Make sure that we are looking at HTML and not other things that
            # are floating around on the internet (such as
            # JavaScript files, CSS, or .PDFs for example)
            if 'text/html' in response.getheader('Content-Type'):
                htmlBytes = response.read()
                # Note that feed() handles Strings well, but not bytes
                # (A change from Python 2.x to Python 3.x)
                htmlString = htmlBytes.decode("utf-8")
                self.search = search
                self.feed(htmlString)
                return self.interesting_data, self.links
            else:
                return "", []
        except HTTPError as error:
            print(url)
            print(error)
            return "", []


def get_links_chileautos(marca, modelo, anho=None):
    baseurl = 'https://www.chileautos.cl'
    query_format = '(C.Marca.{:s}._.Modelo.{:s}.)'.format(marca, modelo)
    query_url = '/autos/busqueda?s={:s}&q={:s}&l={:s}'.format('{:d}',query_format,'{:d}')

    full_url = parse.urljoin(baseurl, query_url)
    all_links = []
    step = 60
    parser = LinkParser()
    data, links = parser.getLinks(full_url.format(0, step), search='/autos/busqueda?s=', base_url=baseurl)
    for start in range(0, len(links) * step, step):

        data, links = parser.getLinks(full_url.format(start, step), search='auto/usado/details/CL-AD', base_url=baseurl)
        all_links = all_links + links
        # for link in links:
        #     print(link)
        # print(len(links))
        # print(data.find('listing-item__details'))
    return all_links


def get_price_chileautos(url):
    baseurl = 'https://www.chileautos.cl'
    parser = LinkParser()
    data, l = parser.getLinks(url, search='', base_url=baseurl)
    if data == "" and l == []:
        return None
    anho = data[1].split(' ')[0]
    data.append('Anho')
    data.append(anho)
    # print(','.join(data))
    try:
        # print(len(data), data)
        tuple_data = [(data[i], data[i + 1]) for i in range(0, len(data), 2)]
        return dict(tuple_data)
    except IndexError:
        print(data)



if __name__ == '__main__':
    marca = 'Volkswagen'
    modelo = 'ESCARABAJO'
    links = get_links_chileautos(marca=marca, modelo=modelo)
    all_dict = []
    count=0
    for link in links:
        print("Processing: {:s}".format(link))
        data_dict = get_price_chileautos(link)
        if data_dict is not None:
            all_dict.append(data_dict)
        # if count == 3:
        #     break
        # else:
        #     count += 1

    all_keys = []
    for d in all_dict:
        for key in d.keys():
            if key not in all_keys:
                all_keys.append(key)
            # else:
            #     print("{:s} repetida".format(key))

    all_data_to_dict = {}
    for key in all_keys:
        all_data_to_dict[key] = []


    for d in all_dict:
        for dkey in d.keys():
            all_data_to_dict[dkey].append(d[dkey])
        max_len = 0
        for akey in all_data_to_dict.keys():
            max_len = max(max_len, len(all_data_to_dict[akey]))
        for akey2 in all_data_to_dict.keys():
            if len(all_data_to_dict[akey2]) < max_len:
                all_data_to_dict[akey2].append("")

    # print(all_data_to_dict)

    df = pd.DataFrame(all_data_to_dict, columns=all_keys)
    df.to_csv('all_{:s}-{:s}.csv'.format(marca, modelo), sep=',')
    # print(df)
