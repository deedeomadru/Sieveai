import { Tag } from 'antd';
import React, { useState } from 'react';
import { Typography } from 'antd';
import { useDispatch, useSelector } from 'react-redux';
import styled from 'styled-components';
import { addProfile } from 'store/ducks';
import { StoreState } from 'store';

const { CheckableTag } = Tag;

const { Title } = Typography;

const tagsData: string[] = ["Software Engineer", "Data Scientist", "Product Manager", "Web Developer", "UX/UI Designer", "DevOps Engineer", "Machine Learning Engineer", "Systems Analyst", "Business Analyst", "Cloud Engineer", "Database Administrator", "Network Engineer", "Full Stack Developer", "Front-End Developer", "Back-End Developer", "Graphic Designer"]

const JobProfileContainer = styled.div`
  .edit-tag {
    padding: 2px 11px;
    margin-bottom: 4px;
    font-size: 13.5px;
  }
`;

const JobProfile: React.FC = () => {
  const suggestedTags = tagsData;
  const profile = useSelector((state: StoreState) => state.stepsContent.profile);
  const [selectedTags, setSelectedTags] = useState<string[]>(profile);
  const dispatch = useDispatch();

  const handleChange = (tag, checked) => {
    const nextSelectedTags = checked ? [...selectedTags, tag] : selectedTags.filter(t => t !== tag);
    setSelectedTags(nextSelectedTags);
    dispatch(addProfile(nextSelectedTags));
  };

  return (
    <JobProfileContainer>
      {suggestedTags && <Title level={5}>Job Profile:</Title>}
      {suggestedTags.map(tag => (
        <CheckableTag
          key={tag}
          className="edit-tag"
          checked={selectedTags.indexOf(tag) > -1}
          onChange={checked => handleChange(tag, checked)}
        >
          {tag}
        </CheckableTag>
      ))}
    </JobProfileContainer>
  );
};
export default JobProfile;
