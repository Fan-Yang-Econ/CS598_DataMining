from setuptools import setup, find_packages


def setup_package():
    setup(  # setup_requires=['six', 'pyscaffold>=2.5a0,<2.6a0'] + sphinx,
        # use_pyscaffold=True,
        name='CS598_DLH',
        packages=find_packages(exclude=('test', 'test_files', 'test_integration')),
        description='',
        author='Fan Yang',
        author_email='yfno1@msn.com',
        scripts=['bin/pre_process_data.py', 'bin/run_model_evaluation_or_fine_tune.py'],
        zip_safe=False,
        include_package_data=True
    )


if __name__ == "__main__":
    setup_package()
