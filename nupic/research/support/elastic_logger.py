#  Numenta Platform for Intelligent Computing (NuPIC)
#  Copyright (C) 2019, Numenta, Inc.  Unless you have an agreement
#  with Numenta, Inc., for a separate license for this software code, the
#  following terms and conditions apply:
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero Public License version 3 as
#  published by the Free Software Foundation.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#  See the GNU Affero Public License for more details.
#
#  You should have received a copy of the GNU Affero Public License
#  along with this program.  If not, see http://www.gnu.org/licenses.
#
#  http://numenta.org/licenses/
#
import os
import re
import subprocess
from datetime import datetime

from elasticsearch import Elasticsearch, helpers
from elasticsearch.client.xpack import SqlClient
from elasticsearch.helpers import BulkIndexError
from pandas import DataFrame
from pandas.io.json.normalize import nested_to_record
from ray.tune.logger import Logger


def create_elastic_client(**kwargs):
    """
    Create and configure :class:`elasticsearch.Elasticsearch` client

    The following environment variables are used to configure the client:

        - **ELASTIC_CLOUD_ID**: The Cloud ID from ElasticCloud. Other host
                                connection params will be ignored. ("cloud_id")
        - **ELASTIC_HOSTS**: List of nodes we should connect to. ("hosts")
        - **ELASTIC_AUTH**: http auth information ("http_auth")

    :param kwargs: Used to override the environment variables or pass extra
                   parameters to the :class:`elasticsearch.Elasticsearch`.
    :type kwargs: dict
    :return: Configured :class:`elasticsearch.Elasticsearch` client
    :rtype: :class:`Elasticsearch`
    """
    hosts = os.environ.get("ELASTIC_HOST")
    hosts = None if hosts is None else hosts.split(",")
    elasticsearch_args = {
        "cloud_id": os.environ.get("ELASTIC_CLOUD_ID"),
        "hosts": hosts,
        "http_auth": os.environ.get("ELASTIC_AUTH")
    }

    # Update elasticsearch client arguments from configuration if present
    elasticsearch_args.update(kwargs)
    return Elasticsearch(**elasticsearch_args)


class ElasticsearchLogger(Logger):
    """
    Elasticsearch Logging interface for `ray.tune`.

    This logger will upload all the results to an elasticsearch index.
    In addition to the regular ray tune log entry, this logger will add the
    the last git commit information and the current `logdir` to the results.

    The following environment variables are used to configure the
    :class:`elasticsearch.Elasticsearch` client:

        - **ELASTIC_CLOUD_ID**: The Cloud ID from ElasticCloud. Other host
                                connection params will be ignored
        - **ELASTIC_HOST**: hostname of the elasticsearch node
        - **ELASTIC_AUTH**: http auth information ('user:password')

    You may override the environment variables or pass extra parameters to the
    :class:`elasticsearch.Elasticsearch` client for the specific experiment
    using the "elasticsearch_client" configuration key.

    The elasticsearch index name is based on the current results root path. You
    may override this behavior and use a specific index name for your experiment
    using the configuration key `elasticsearch_index`.
    """

    def _init(self):
        elasticsearch_args = self.config.get("elasticsearch_client", {})
        self.client = create_elastic_client(**elasticsearch_args)

        # Save git information
        self.git_remote = subprocess.check_output(
            ["git", "ls-remote", "--get-url"]).decode("ascii").strip()
        self.git_branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode("ascii").strip()
        self.git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"]).decode("ascii").strip()
        self.git_user = subprocess.check_output(
            ["git", "log", "-n", "1", "--pretty=format:%an"]).decode("ascii").strip()

        # Check for elasticsearch index name in configuration
        index_name = self.config.get("elasticsearch_index")
        if index_name is None:
            # Create default index name based on log path and git repo name
            git_root = subprocess.check_output(
                ["git", "rev-parse", "--show-toplevel"]).decode("ascii").strip()
            repo_name = os.path.basename(self.git_remote).rstrip(".git")
            path_name = os.path.relpath(self.config["path"], git_root)
            index_name = os.path.join(repo_name, path_name)

            # slugify index name
            index_name = re.sub(r"[\W_]+", "-", index_name)

        self.index_name = index_name

        self.logdir = os.path.basename(self.logdir)
        self.experiment_name = self.config["name"]
        self.buffer = []

    def on_result(self, result):
        """Given a result, appends it to the existing log."""
        log_entry = {
            "git": {
                "remote": self.git_remote,
                "branch": self.git_branch,
                "sha": self.git_sha,
                "user": self.git_user
            },
            "logdir": self.logdir
        }
        # Convert timestamp to ISO-8601
        timestamp = result["timestamp"]
        result["timestamp"] = datetime.utcfromtimestamp(timestamp).isoformat()

        log_entry.update(result)
        self.buffer.append(log_entry)

    def close(self):
        self.flush()

    def flush(self):
        if len(self.buffer) > 0:
            results = helpers.parallel_bulk(client=self.client,
                                            actions=self.buffer,
                                            index=self.index_name,
                                            doc_type=self.experiment_name)
            errors = [status for success, status in results if not success]
            if errors:
                raise BulkIndexError("{} document(s) failed to index.".
                                     format(len(errors)), errors)

            self.buffer.clear()


def elastic_dsl(client, dsl, index, **kwargs):
    """
    Sends DSL query to elasticsearch and returns the results as a
    :class:`pandas.DataFrame`.

    :param client: Configured elasticseach client. See :func:`create_elastic_client`
    :type client: :class:`elasticseach.Elasticsearch`
    :param dsl: Elasticsearch DSL query statement
                See https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl.html  # noqa: E501
    :type dsl: str
    :param index: Index pattern. Usually the same as 'from' part of the SQL
                  See https://www.elastic.co/guide/en/elasticsearch/reference/current/multi-index.html # noqa: E501
    :type index: str

    :param kwargs: Any additional keyword arguments will be passed to the initial
                   :meth:`elasticsearch.Elasticsearch.search` call
    :type kwargs: dict

    :return: results as a :class:`pandas.DataFrame`.
    :rtype: :class:`pandas.DataFrame`
    """
    response = helpers.scan(client=client, query=dsl, index=index, **kwargs)
    data = []
    for row in response:
        # Normalize nested dicts in '_source' such as 'config' or 'git'
        source = nested_to_record(row["_source"]) if "_source" in row else {}

        # Squeeze scalar fields returned as arrays in the response by the search API
        fields = row.get("fields", {})
        fields = {k: v[0] if len(v) == 1 else v for k, v in fields.items()}

        data.append({
            "_index": row["_index"],
            "_type": row["_type"],
            **fields,
            **source,
        })
    return DataFrame(data)


def elastic_sql(client, sql, index, **kwargs):
    """
    Sends SQL query to elasticsearch and returns the results as a
    :class:`pandas.DataFrame`.

    :param client: Configured elasticseach client. See :func:`create_elastic_client`
    :type client: :class:`elasticseach.Elasticsearch`
    :param sql: Elasticsearch SQL query statement
                See https://www.elastic.co/guide/en/elasticsearch/reference/current/xpack-sql.html  # noqa: E501
    :type sql: str
    :param index: Index pattern. Usually the same as 'from' part of the SQL
                  See https://www.elastic.co/guide/en/elasticsearch/reference/current/multi-index.html
    :type index: str
    :param kwargs: Any additional keyword arguments will be passed to the initial
                   :meth:`elasticsearch.Elasticsearch.search` call

    :type kwargs: dict
    :return: results as a :class:`pandas.DataFrame`.
    :rtype: :class:`pandas.DataFrame`
    """
    sql_client = SqlClient(client)
    # FIXME: SQL API does not support arrays. See https://github.com/elastic/elasticsearch/issues/33204  # noqa: E501
    # For now we translate the SQL into elasticsearch DSL query and use the
    # 'search' API to fetch the results
    dsl = sql_client.translate({"query": sql})

    # Ignore score
    if "query" in dsl:
        query = dsl["query"]
        dsl["query"] = {"constant_score": {"filter": query}}

    return elastic_dsl(client, dsl, index, **kwargs)
