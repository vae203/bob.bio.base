#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

from .database import DummyDatabase
from bob.bio.base.database.file import BioFile

class DummyDatabaseMissingFiles(DummyDatabase):

    def __init__(self):

        # call base class constructor with useful parameters
        super(DummyDatabaseMissingFiles, self).__init__()
        
    def objects(self, groups=None, protocol=None, purposes=None, model_ids=None, **kwargs):
        biofiles = self._make_bio(self._db.objects(model_ids, groups, purposes, protocol, **kwargs))
        if groups=="world":
            fake_biofiles = [BioFile(client_id='fake-client-id', path='fake-path', file_id='fake-id')]
            return biofiles + fake_biofiles
        return biofiles


database = DummyDatabaseMissingFiles()
