'''
Run as: `scrapy runspiders books.py`
'''

import logging
import re
from pathlib import Path

import scrapy
from scrapy.http import Request


class BookScrape(scrapy.Spider):
    name = 'book_scrape'

    allowed_domains = ['programmer-books.com', 'download.programmer-books.com', ]
    start_urls = ['https://www.programmer-books.com/category-sitemap.xml']
    base_path = Path(f'../../data/books')

    def parse(self, response):
        for href in re.finditer(r'<loc>([^<]+)</loc>', response.body_as_unicode()):
            href = href.group(1)
            logging.critical(f'\n\nHref: {href}\n\n')
            url = response.urljoin(href)
            category = re.search(r'/category/([a-zA-Z-]+)/', href).group(1)
            logging.critical(f'\n\nCat: {category}\n\n')

            meta = {'subfolder': category}
            yield Request(
                url=url,
                callback=self.parse_category,
                meta=meta,
            )

    def parse_category(self, response):
        for href in response.css(f'h3 a::attr(href)').extract():
            url = response.urljoin(href)
            yield Request(
                url=url,
                callback=self.parse_book_page,
                meta=response.meta
            )

        for next_page in response.css('.page-nav'):
            for href in next_page.css('a'):
                next_href = href.css('.td-icon-menu-right').extract_first()
                if next_href:
                    url = href.css('::attr(href)').extract_first()
                    yield scrapy.Request(
                        url=response.urljoin(url),
                        callback=self.parse_category,
                        meta=response.meta,
                    )

    def parse_book_page(self, response):
        download_class = '.s_pdf_download_link'
        for href in response.css(f'{download_class}::attr(href)').extract():
            url = response.urljoin(href)
            url = re.sub(r'\.com\?', r'.com/?', url)  # ISSUE: https://github.com/scrapy/scrapy/issues/3540
            yield Request(
                url=url,
                callback=self.parse_download_page,
                meta=response.meta,
            )

    def parse_download_page(self, response):
        for href in response.css('a::attr(href)').extract():
            url = response.urljoin(href)

            # subfolder = response.meta.to_get('subfolder', '')     # older scrapy version
            subfolder = response.meta.get('subfolder', '')
            dir_path = Path(f'data/books') / subfolder
            dir_path.mkdir(parents=True, exist_ok=True)
            path = dir_path / response.url.split('/')[-1]
            if path.exists():
                logging.info(f'\n\nSkipping already existing {path}\n\n')
                continue

            yield Request(
                url=url,
                callback=self.save_pdf,
                meta=response.meta,
            )

    def save_pdf(self, response):
        # subfolder = response.meta.to_get('subfolder', '')     # older scrapy version
        subfolder = response.meta.get('subfolder', '')
        dir_path = self.base_path / subfolder
        dir_path.mkdir(parents=True, exist_ok=True)
        path = dir_path / response.url.split('/')[-1]
        self.logger.info(f'\n\nSaving PDF {path}\n\n')
        # self.logger.info(f'\n\nSaving to subfolder {subfolder}\n\n')
        if path.exists():
            logging.info(f'\n\nLast minute skipping already existing {path}\n'
                         f'\tThis means the PDF was already downloaded, which is highly inefficient\n')
            return

        with open(path, 'wb') as f:
            f.write(response.body)
