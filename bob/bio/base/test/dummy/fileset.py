from bob.bio.base.database import ZTBioDatabase, BioFileSet, BioFile
from bob.bio.base.test.utils import atnt_database_directory


class DummyDatabase(ZTBioDatabase):

    def __init__(self):
        # call base class constructor with useful parameters
        super(DummyDatabase, self).__init__(
            name='test_fileset',
            original_directory=atnt_database_directory(),
            original_extension='.pgm',
            check_original_files_for_existence=True,
            training_depends_on_protocol=False,
            models_depend_on_protocol=False
        )
        import bob.db.atnt
        self.__db = bob.db.atnt.Database()

    def uses_probe_file_sets(self):
        return True

    def probe_file_sets(self, model_id=None, group='dev'):
        """Returns the list of probe File objects (for the given model id, if given)."""
        # import ipdb; ipdb.set_trace()
        files = self.arrange_by_client(self.sort(self.objects(protocol=None, groups=group, purposes='probe')))
        # arrange files by clients
        file_sets = []
        for client_files in files:
            # convert into our File objects (so that they are tested as well)
            our_files = [BioFile(f.client_id, f.path, f.id) for f in client_files]
            # generate file set for each client
            file_set = BioFileSet(our_files[0].client_id, our_files)
            file_sets.append(file_set)
        return file_sets

    def model_ids_with_protocol(self, groups=None, protocol=None, **kwargs):
        return self.__db.model_ids(groups, protocol)

    def objects(self, groups=None, protocol=None, purposes=None, model_ids=None, **kwargs):
        return self.__db.objects(model_ids, groups, purposes, protocol, **kwargs)

    def tobjects(self, groups=None, protocol=None, model_ids=None, **kwargs):
        return []

    def zobjects(self, groups=None, protocol=None, **kwargs):
        return []

    def tmodel_ids_with_protocol(self, protocol=None, groups=None, **kwargs):
        return self.__db.model_ids(groups)

    def t_enroll_files(self, t_model_id, group='dev'):
        return self.enroll_files(t_model_id, group)

    def z_probe_files(self, group='dev'):
        return self.probe_files(None, group)

    def z_probe_file_sets(self, group='dev'):
        return self.probe_file_sets(None, group)

database = DummyDatabase()
