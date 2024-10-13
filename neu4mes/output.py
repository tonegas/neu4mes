from pprint import pformat

from neu4mes.relation import Stream
from neu4mes.utilis import check

from neu4mes import LOG_LEVEL
from neu4mes.logger import logging
log = logging.getLogger(__name__)
log.setLevel(max(logging.ERROR, LOG_LEVEL))

class Output(Stream):
    def __init__(self, name, relation):
        super().__init__(name, relation.json, relation.dim)
        log.debug(f"Output {name}")
        self.json['Outputs'][name] = {}
        self.json['Outputs'][name] = relation.name
        log.debug("\n"+pformat(self.json))

    def closedLoop(self, obj):
        check(False, TypeError,
              f"The {self} must be a Stream and not a {type(self)}.")

    def connect(self, obj):
        check(False, TypeError,
              f"The {self} must be a Stream and not a {type(self)}.")