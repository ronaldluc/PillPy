'''
Run as: `python -m scrapy runspider benu.py -o data/doplnky.csv -t csv` to save it to a cvs
'''

import logging
import re
from pathlib import Path

import scrapy
from scrapy.http import Request


class BenuScrape(scrapy.Spider):
    name = 'book_scrape'

    allowed_domains = ['www.benu.cz', ]
    start_urls = ['https://www.benu.cz/volne-prodejne-leky']
    start_urls = ['https://www.benu.cz/vyziva-a-doplnky-stravy']
    base_path = Path(f'/data/imgs')

    def parse(self, response):
        print(response.url)
        #print(response.xpath('//div[@class="spc"]//span[@class="name"]/text()').extract())
        #print(response.xpath('//div[@class="spc"]//img/@src').extract())
        for card in response.xpath(f'//div[@class="spc"]'):
            print(card)
            product = {}
            product['name'] = card.xpath('.//span[@class="name"]/text()').extract_first()
            product['path'] = [x for x in card.xpath('.//img/@src').extract() if 'http' in x]
            print(product)
            if len(product['path']) > 0:
                product['path'] = product['path'][0]
                yield product
                
        url = response.xpath('//a[contains(@class, "next")]/@href').extract_first()
        url = response.urljoin(url)
        yield Request(
            url=url,
            callback=self.parse,
            meta=response.meta
        )
