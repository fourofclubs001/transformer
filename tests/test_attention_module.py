from test_base import *
from src.AttentionModule import *
import torch

class AttentionModuleTest(BaseTest):

    def setUp(self):

        super().setUp()

        self.onesTensor = torch.ones((1,self.sequenceLenght, self.modelDimension))
        
        self.attentionModule = AttentionModule(self.modelDimension)

        self.query = torch.ones((2, self.querySequenceLenght, self.modelDimension))
        self.key = torch.ones((2, self.keySequenceLenght, self.modelDimension))
        self.value = torch.ones((2, self.keySequenceLenght, self.modelDimension))

    def test_can_calculate_compatibility(self):

        query = self.onesTensor.clone()
        key = self.onesTensor.clone()

        compatibility = self.attentionModule.calculateCompatibility(query, key)

        expectedCompatibility = torch.ones((1, self.sequenceLenght, self.sequenceLenght))*self.modelDimension

        self.assertTrue(torch.eq(compatibility, expectedCompatibility).all())

    def test_can_calculate_compatibility_with_different_sequence_leght(self):

        query = self.onesTensor.clone()
        key = torch.ones((1, self.sequenceLenght+1, self.modelDimension))

        compatibility = self.attentionModule.calculateCompatibility(query, key)

        expectedCompatibility = torch.ones((1, self.sequenceLenght, self.sequenceLenght+1))*self.modelDimension

        self.assertTrue(torch.eq(compatibility, expectedCompatibility).all())

    def test_can_calculate_compatibility_with_greater_batch_size(self):

        compatibility = self.attentionModule.calculateCompatibility(self.query, self.key)

        expectedCompatibility = torch.ones((2, self.querySequenceLenght, self.keySequenceLenght))*self.modelDimension

        self.assertTrue(torch.eq(compatibility, expectedCompatibility).all())

    def test_can_scale_compatibility_matrix(self):

        compatibility = torch.ones((2, self.querySequenceLenght, self.keySequenceLenght))
        expectedCompatibility = torch.div(compatibility, torch.sqrt(torch.tensor(self.querySequenceLenght)))

        compatibility = self.attentionModule.scaleCompatibility(compatibility)

        self.assertTrue(torch.eq(compatibility, expectedCompatibility).all())

    def test_can_apply_softmax_over_compatibility(self):

        compatibility = torch.ones((2, self.querySequenceLenght, self.keySequenceLenght))

        compatibility = self.attentionModule.softmaxCompatibility(compatibility)

        sumOne = torch.sum(compatibility, dim=2)
        expected = torch.ones((2, self.querySequenceLenght))

        self.assertTrue(torch.eq(sumOne, expected).all())

    def test_can_calculate_output_tokens_using_values(self):

        compatibility = torch.ones((2, self.querySequenceLenght, self.keySequenceLenght))

        output = self.attentionModule.calculateOutput(compatibility, self.value)

        expected = compatibility @ self.value

        self.assertTrue(torch.eq(output, expected).all())

    def test_can_do_pass_foward(self):

        output = self.attentionModule(self.query, self.key, self.value)

        expected = torch.ones((2, self.querySequenceLenght, self.modelDimension))

        self.assert_equal_dimensions(output, expected)

    def test_can_select_model_dimension_token_lenght(self):

        modelDimension = 3
        attentionBlock = AttentionModule(modelDimension)

        query = torch.ones((1, self.sequenceLenght, modelDimension))
        key = torch.ones((1, self.sequenceLenght, modelDimension))
        value = torch.ones((1, self.sequenceLenght, modelDimension))

        output = attentionBlock(query, key, value)
        expected = torch.ones((1, self.sequenceLenght, modelDimension))

        self.assert_equal_dimensions(output, expected)

    def test_can_apply_mask(self):

        attentionModule = AttentionModule(self.modelDimension, applyMask = True)

        compatibility = torch.ones((2, self.querySequenceLenght, self.keySequenceLenght))

        compatibility = attentionModule.softmaxCompatibility(compatibility)

        ones = torch.ones_like(compatibility)
        neutralForMask = torch.tril(ones)
        appliedMask = neutralForMask * compatibility

        self.assertTrue(torch.eq(appliedMask, compatibility).all())

    def assert_raise_error_calculate_compatibility(self,
                                                   query: torch.Tensor, 
                                                   key: torch.Tensor, 
                                                   exception: Exception, 
                                                   errorMessage: str):

        with self.assertRaises(exception) as error:

            self.attentionModule.calculateCompatibility(query, key)

        self.assertEqual(error.exception.args[0], errorMessage)

    def test_raise_exception_when_different_token_size_on_compatibility(self): 

        query = self.onesTensor
        key = torch.ones((1, self.sequenceLenght, self.modelDimension+1))

        self.assert_raise_error_calculate_compatibility(query, key, 
                                                        CannotUseDifferentQueryAndKeyTokenLenght, 
                                                        CANNOT_USE_DIFFERENT_QUERY_AND_KEY_TOKEN_LENGHT_ERROR_MSG)

    def test_raise_exception_when_different_number_of_batch_on_compatibility(self):

        query = self.onesTensor
        key = torch.ones((2, self.sequenceLenght, self.modelDimension))

        self.assert_raise_error_calculate_compatibility(query, key, 
                                                        CannotUseDifferentQueryAndKeyBatchLenght, 
                                                        CANNOT_USE_DIFFERENT_QUERY_AND_KEY_BATCH_LENGHT_ERROR_MSG)
        
    def test_raise_exception_when_forward_with_different_key_and_value_sequence_lenght(self):

        query = torch.ones((1, self.sequenceLenght, self.modelDimension))
        key = torch.ones((1, self.sequenceLenght, self.modelDimension))
        value = torch.ones((1, self.sequenceLenght + 1, self.modelDimension))

        with self.assertRaises(CannotForwardWithDifferentKeyValueSequenceLenght) as error:

            self.attentionModule(query, key, value)

        self.assertEqual(error.exception.args[0], CANNOT_FORWARD_WITH_DIFFERENT_KEY_AND_VALUE_SEQUENCE_LENGHT)

    def test_raise_exception_when_forward_query_token_lenght_does_not_match_model_dimension(self):

        with self.assertRaises(QueryKeyValueTokenLenghtMustMatchModelDimension) as error:

            query = torch.ones((1, self.sequenceLenght, self.modelDimension+1))
            key = torch.ones((1, self.sequenceLenght, self.modelDimension))
            value = torch.ones((1, self.sequenceLenght, self.modelDimension))

            self.attentionModule(query, key, value)

        self.assertEqual(error.exception.args[0], QUERY_KEY__VALUE_TOKEN_LENGHT_MUST_MODEL_DIMENSION)

    def test_raise_exception_when_forward_key_does_not_match_initial_token_lenght(self):

        with self.assertRaises(QueryKeyValueTokenLenghtMustMatchModelDimension) as error:

            query = torch.ones((1, self.sequenceLenght, self.modelDimension))
            key = torch.ones((1, self.sequenceLenght, self.modelDimension+1))
            value = torch.ones((1, self.sequenceLenght, self.modelDimension))

            self.attentionModule(query, key, value)

        self.assertEqual(error.exception.args[0], QUERY_KEY__VALUE_TOKEN_LENGHT_MUST_MODEL_DIMENSION)

    def test_raise_exception_when_forward_value_does_not_match_initial_token_lenght(self):

        with self.assertRaises(QueryKeyValueTokenLenghtMustMatchModelDimension) as error:

            query = torch.ones((1, self.sequenceLenght, self.modelDimension))
            key = torch.ones((1, self.sequenceLenght, self.modelDimension))
            value = torch.ones((1, self.sequenceLenght, self.modelDimension+1))

            self.attentionModule(query, key, value)

        self.assertEqual(error.exception.args[0], QUERY_KEY__VALUE_TOKEN_LENGHT_MUST_MODEL_DIMENSION)

    def test_can_select_device_for_forward(self):

        device = torch.device('cuda')
        
        self.query = self.query.to(device)
        self.key = self.key.to(device)
        self.value = self.value.to(device)

        self.attentionModule.to(device)

        output = self.attentionModule(self.query, self.key, self.value)

        self.assertEqual(output.device.type, device.type)