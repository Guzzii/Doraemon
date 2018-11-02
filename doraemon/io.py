import json
import logging
import http.client

import numpy as np
import pandas as pd
import cx_Oracle as cxo

logger = logging.getLogger(__name__)


class OracleHook(object):
    """
    Oracle database wrapper class
    """
    def __init__(self, user, password, host, port, service_name, query=None):
        self._query = query
        self._dsn = cxo.makedsn(host=host, port=port, service_name=service_name)
        self._con = cxo.connect(user=user, password=password, dsn=self._dsn)
        self._cur = self._con.cursor()

    @staticmethod
    def parse_query(file):
        with open(file, 'r') as f:
            return f.read().splitlines()

    @property
    def query(self):
        return self._query

    @query.setter
    def query(self, value):
        if not isinstance(value, str):
            logger.debug(value)
            raise TypeError('Query should be a str or path')

        if '.sql' in val:
            self._query = self.parse_query(value)
        else:
            self._query = value

    def _fetch(self, mode='all'):
        if query is None:
            raise ValueError('Query cannot be none.')

        logger.info('Querying data...')
        self._cur.execute(self._query)

        if mode == 'one':
            rows = self._cur.fetchone()
        elif mode == 'all':
            rows = self._cur.fetchall()
        else:
            raise NotImplementedError

        cols = [x[0] for x in self._cur.description]
        logger.info('Finished')

        if isinstance(rows, tuple):
            rows = [rows]

        return pd.DataFrame(rows, columns=cols)

    def one_to_df(self):
        return self._fetch('one')

    def all_to_df(self):
        return self._fetch('all')

    def close(self):
        self._cur.close()
        self._con.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class APIHook(object):
    """
    API wrapper class
    """
    def __init__(self, host=None, url=None, headers=None):
        self._host = host
        self._url = url
        self._headers = {'Accept': 'application/json',
                         'Content-Type': 'application/json',
                         'Cache-Control': 'no-cache'}

        self._conn = http.client.HTTPConnection(self._host)

    def request(self, req_type, payload):
        self._conn.request(req_type, url=self._url, body=payload, headers=self._headers)
        res = self._conn.getresponse().read().decode("utf-8")
        return json.loads(res)

    def close(self):
        self._conn.close()
