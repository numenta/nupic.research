#  Numenta Platform for Intelligent Computing (NuPIC)
#  Copyright (C) 2020, Numenta, Inc.  Unless you have an agreement
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

from urllib.parse import urlparse

import boto3


def s3_create_presigned_url(s3url, expiration=3600):
    """Generate a presigned URL to share an S3 object

    :param s3url: string
    :param expiration: Time in seconds for the pre-signed URL to remain valid
    :return: Pre-signed URL as string
    """

    components = urlparse(s3url)
    bucket_name = components.hostname
    object_name = components.path[1:]

    # Generate a pre-signed URL for the S3 object
    s3_client = boto3.client("s3")
    response = s3_client.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket_name, "Key": object_name},
        ExpiresIn=expiration,
    )
    # The response contains the pre-signed URL
    return response
