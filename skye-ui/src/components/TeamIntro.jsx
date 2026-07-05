import React, { useEffect, useState, useRef } from 'react';
import './TeamIntro.css';

const teamMembers = [
  {
    name: 'Deepakkumar Senthilmurugan',
    role: 'Team Lead & ML Architecture',
    icon: 'hub',
    seed: 'Deepakkumar',
  },
  {
    name: 'Hari R Krishna',
    role: 'Edge Deployment & Optimization',
    icon: 'memory',
    seed: 'Hari',
  },
  {
    name: 'Subiksha Maniselvam',
    role: 'Sensor Simulation & Data Engineering',
    icon: 'sensors',
    seed: 'Subiksha',
  },
  {
    name: 'Yuva Shree G',
    role: 'RUL Prediction & Model Training',
    icon: 'model_training',
    seed: 'Yuva',
  },
  {
    name: 'Harsini R G',
    role: 'Dashboard & Federated Learning',
    icon: 'dashboard',
    seed: 'Harsini',
  },
];

const TeamIntro = () => {
  const [stage, setStage] = useState('initial'); 
  const [taglineText, setTaglineText] = useState('');
  const tagline = 'Intelligence at altitude.';
  const sectionRef = useRef(null);
  
  useEffect(() => {
    
    let index = 0;
    const typeInterval = setInterval(() => {
      setTaglineText(tagline.slice(0, index + 1));
      index++;
      if (index >= tagline.length) {
        clearInterval(typeInterval);
        
        
        setTimeout(() => {
          setStage('header-moved');
        }, 1000);
      }
    }, 50);

    return () => clearInterval(typeInterval);
  }, []);

  useEffect(() => {
    if (stage === 'header-moved') {
      
      const observer = new IntersectionObserver(
        (entries) => {
          entries.forEach((entry) => {
            if (entry.isIntersecting) {
              setStage('cards-visible');
              observer.disconnect();
            }
          });
        },
        { threshold: 0.2 }
      );
      
      if (sectionRef.current) {
        observer.observe(sectionRef.current);
      }
      return () => observer.disconnect();
    }
  }, [stage]);

  return (
    <div className="team-intro-container" ref={sectionRef}>
      {}
      <div className="particles-overlay"></div>
      <div className="grid-overlay"></div>

      {}
      <div className={`intro-header ${stage !== 'initial' ? 'nav-mode' : ''}`}>
        <h1 className="skye-logo">SKYE</h1>
        <p className="skye-tagline">{taglineText}</p>
      </div>

      {}
      <div className={`team-section ${stage === 'cards-visible' ? 'visible' : ''}`}>
        <div className="team-headline-wrapper">
          <h2 className="team-headline">TEAM SKYPULSE</h2>
          <div className="hud-line"></div>
        </div>

        <div className="cards-grid">
          {teamMembers.map((member, index) => (
            <div 
              className="team-card" 
              key={member.name}
              style={{ animationDelay: `${index * 150}ms` }}
            >
              <div className="avatar-wrapper">
                <img 
                  src={`https://api.dicebear.com/7.x/notionists/svg?seed=${member.seed}&backgroundColor=transparent`}
                  alt={member.name}
                  className="avatar-image"
                />
                <div className="glow-ring"></div>
              </div>
              <h3 className="member-name">{member.name}</h3>
              <p className="member-role">{member.role}</p>
              <div className="icon-wrapper">
                <span className="material-symbols-outlined member-icon">
                  {member.icon}
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default TeamIntro;
